"""Champion / challenger lifecycle on top of MLflow model-registry aliases.

Lifecycle of a new model version::

    trained -> offline gate --(no champion yet)--> champion          (bootstrap)
                           --(beats champion)----> challenger        (gets X% traffic, A/B)
                           --(does not beat)-----> rejected          (tagged, no alias)
    challenger --(recsys promote)--> champion; old champion -> previous_champion
    champion   --(recsys rollback)-> previous_champion (instant: the API re-resolves aliases)
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

CHAMPION = "champion"
CHALLENGER = "challenger"
PREVIOUS = "previous_champion"
SHADOW = "shadow"  # scored in the background on live requests, never served
LOWER_IS_BETTER = {"rmse", "mae"}

Action = Literal["bootstrap_champion", "challenger", "rejected"]


@dataclass(frozen=True)
class GateDecision:
    action: Action
    metric: str
    candidate_value: float
    champion_value: float | None
    min_relative_improvement: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def decide(
    metric: str,
    candidate_value: float,
    champion_value: float | None,
    min_relative_improvement: float = 0.0,
    bootstrap_champion: bool = True,
) -> GateDecision:
    """Offline promotion gate: the candidate must *strictly* beat the champion."""
    if champion_value is None:
        action: Action = "bootstrap_champion" if bootstrap_champion else "challenger"
        return GateDecision(
            action, metric, candidate_value, None, min_relative_improvement, "no champion yet"
        )
    lower = metric in LOWER_IS_BETTER
    if lower:
        target = champion_value * (1.0 - min_relative_improvement)
        better = candidate_value < target
    else:
        target = champion_value * (1.0 + min_relative_improvement)
        better = candidate_value > target
    reason = (
        f"{metric}: candidate={candidate_value:.5f} vs champion={champion_value:.5f} "
        f"(needs {'<' if lower else '>'} {target:.5f})"
    )
    return GateDecision(
        "challenger" if better else "rejected",
        metric,
        candidate_value,
        champion_value,
        min_relative_improvement,
        reason,
    )


class ModelRegistry:
    """Thin wrapper over ``MlflowClient`` alias operations."""

    def __init__(self, name: str) -> None:
        from mlflow import MlflowClient

        self.name = name
        self.client = MlflowClient()

    def version_for(self, alias: str) -> str | None:
        from mlflow.exceptions import MlflowException

        try:
            return str(self.client.get_model_version_by_alias(self.name, alias).version)
        except MlflowException:
            return None

    def aliases(self) -> dict[str, str | None]:
        aliases = (CHAMPION, CHALLENGER, PREVIOUS, SHADOW)
        return {alias: self.version_for(alias) for alias in aliases}

    def _tag(self, version: str, tags: dict[str, object]) -> None:
        for key, value in tags.items():
            self.client.set_model_version_tag(self.name, version, key, str(value))

    def apply_decision(self, version: str, decision: GateDecision) -> None:
        self._tag(
            version,
            {
                "gate_action": decision.action,
                "gate_metric": decision.metric,
                "gate_candidate_value": f"{decision.candidate_value:.6f}",
                "gate_champion_value": (
                    "none" if decision.champion_value is None else f"{decision.champion_value:.6f}"
                ),
                "gate_reason": decision.reason,
            },
        )
        if decision.action == "bootstrap_champion":
            self.promote(version)
        elif decision.action == "challenger":
            self.client.set_registered_model_alias(self.name, CHALLENGER, version)
        logger.info("model %s v%s -> %s (%s)", self.name, version, decision.action, decision.reason)

    def promote(self, version: str) -> None:
        current = self.version_for(CHAMPION)
        if current == str(version):
            return
        if current is not None:
            self.client.set_registered_model_alias(self.name, PREVIOUS, current)
        self.client.set_registered_model_alias(self.name, CHAMPION, str(version))
        if self.version_for(CHALLENGER) == str(version):
            self.client.delete_registered_model_alias(self.name, CHALLENGER)
        logger.info("promoted %s v%s to champion (previous=%s)", self.name, version, current)

    def rollback(self) -> str:
        previous = self.version_for(PREVIOUS)
        if previous is None:
            raise RuntimeError("no previous_champion alias to roll back to")
        current = self.version_for(CHAMPION)
        self.client.set_registered_model_alias(self.name, CHAMPION, previous)
        if current is not None:
            self.client.set_registered_model_alias(self.name, PREVIOUS, current)
        logger.info("rolled back champion %s -> %s", current, previous)
        return previous

    def set_shadow(self, version: str | None) -> None:
        """Point the shadow alias at ``version`` (or remove it with ``None``).

        The gate never assigns this alias: shadowing is a deliberate choice, typically for a
        candidate that lost offline but whose serving cost or behaviour needs measuring.
        """
        if version is None:
            if self.version_for(SHADOW) is not None:
                self.client.delete_registered_model_alias(self.name, SHADOW)
            logger.info("cleared the shadow alias of %s", self.name)
            return
        self.client.set_registered_model_alias(self.name, SHADOW, str(version))
        logger.info("model %s v%s -> shadow", self.name, version)

    def download_package(self, version: str, dest: Path) -> Path:
        """Copy the model package of ``version`` to ``dest`` (replacing it)."""
        import mlflow

        with tempfile.TemporaryDirectory() as tmp:
            local = Path(
                mlflow.artifacts.download_artifacts(
                    artifact_uri=f"models:/{self.name}/{version}", dst_path=tmp
                )
            )
            package = next(local.rglob("package.json")).parent
            if dest.exists():
                shutil.rmtree(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(package, dest)
        return dest

    def export_aliases(self, artifacts_dir: Path) -> dict[str, str | None]:
        """Mirror alias packages to ``artifacts/models/<alias>`` for local serving."""
        exported: dict[str, str | None] = {}
        for alias in (CHAMPION, CHALLENGER, SHADOW):
            version = self.version_for(alias)
            dest = Path(artifacts_dir) / "models" / alias
            if version is None:
                if dest.exists():
                    shutil.rmtree(dest)
                exported[alias] = None
                continue
            self.download_package(version, dest)
            (dest / "REGISTRY_VERSION").write_text(version, encoding="utf-8")
            exported[alias] = version
        return exported
