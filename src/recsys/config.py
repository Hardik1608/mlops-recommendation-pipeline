"""Runtime settings (environment variables) and training configuration (YAML).

* ``Settings`` - 12-factor runtime config read from ``RECSYS_*`` environment variables
  (and an optional ``.env`` file). Used by every service: API, stream processor, jobs.
* ``TrainConfig`` - versioned, reviewable hyper-parameters in ``configs/<dataset>.yaml``.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CacheBackend(StrEnum):
    MEMORY = "memory"
    REDIS = "redis"
    NONE = "none"


class FeatureBackend(StrEnum):
    MEMORY = "memory"
    FEAST = "feast"


class ModelSource(StrEnum):
    LOCAL = "local"
    MLFLOW = "mlflow"


class Settings(BaseSettings):
    """Runtime configuration. Every field can be overridden with ``RECSYS_<FIELD>``."""

    model_config = SettingsConfigDict(
        env_prefix="RECSYS_", env_file=".env", extra="ignore", protected_namespaces=()
    )

    # --- paths -------------------------------------------------------------------------
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")
    configs_dir: Path = Path("configs")
    dataset: str = "ml-latest-small"

    # --- experiment tracking / registry --------------------------------------------------
    mlflow_tracking_uri: str = Field(
        default="sqlite:///mlruns/mlflow.db",
        validation_alias=AliasChoices("RECSYS_MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_URI"),
    )
    mlflow_artifact_root: str | None = None  # defaults to <cwd>/mlruns/artifacts for sqlite
    mlflow_experiment: str = "recsys-training"
    registered_model_name: str = "recsys-ranker"

    # --- serving ---------------------------------------------------------------------------
    model_source: ModelSource = ModelSource.LOCAL
    champion_model_dir: Path = Path("artifacts/models/champion")
    challenger_model_dir: Path | None = None
    model_refresh_seconds: float = 0.0  # >0: poll the registry and hot-swap on alias change
    challenger_traffic: float = Field(default=0.1, ge=0.0, le=1.0)
    ab_salt: str = "recsys-ab-v1"
    # Shadow model (registry alias "shadow"): scored in the background for this share of
    # uncached requests to measure its latency and agreement; its output is never served.
    shadow_model_dir: Path | None = None
    shadow_traffic: float = Field(default=0.2, ge=0.0, le=1.0)
    default_k: int = Field(default=10, ge=1)
    max_k: int = Field(default=100, ge=1)
    # Single-request scoring is a small GEMV: BLAS thread pools add sync overhead and
    # oversubscribe CPUs when several pods share a node. 1 thread halved scoring latency here.
    blas_threads: int = Field(default=1, ge=1)

    cache_backend: CacheBackend = CacheBackend.MEMORY
    cache_ttl_seconds: int = Field(default=300, ge=1)
    cache_max_entries: int = Field(default=100_000, ge=1)
    redis_url: str = "redis://localhost:6379/0"
    redis_timeout_seconds: float = 0.05
    cache_cooldown_seconds: float = 30.0

    # --- features --------------------------------------------------------------------------
    feature_backend: FeatureBackend = FeatureBackend.MEMORY
    feast_repo_path: Path = Path("feature_repo")
    feast_yaml: str = "feature_store.yaml"  # feature_store.redis.yaml in docker / k8s
    recent_items_max: int = Field(default=20, ge=1)
    # Run online-feature lookups in a worker thread (right for network-bound stores such as
    # Redis). For in-process stores (Feast + SQLite) the work is GIL-bound, so a thread only
    # adds contention: set False to call inline.
    feature_store_in_thread: bool = True

    # --- streaming -------------------------------------------------------------------------
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic: str = "ratings.v1"
    kafka_dlq_topic: str = "ratings.v1.dlq"
    kafka_group_id: str = "recsys-feature-writer"
    sink_flush_events: int = Field(default=5_000, ge=1)
    sink_flush_seconds: float = 30.0

    # --- monitoring ------------------------------------------------------------------------
    pushgateway_url: str | None = None
    drift_psi_threshold: float = 0.2

    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"

    # --- derived paths ---------------------------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    def processed_dir(self, dataset: str | None = None) -> Path:
        return self.data_dir / "processed" / (dataset or self.dataset)

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def events_dir(self) -> Path:
        return self.data_dir / "events"

    def train_config_path(self, dataset: str | None = None) -> Path:
        return self.configs_dir / f"{dataset or self.dataset}.yaml"


# ------------------------------------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------------------------------------
ModelType = Literal["popularity", "explicit_als", "implicit_als", "sasrec"]


class SplitConfig(BaseModel):
    val_fraction: float = Field(default=0.1, ge=0.0, lt=1.0)
    test_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)


class EvaluationConfig(BaseModel):
    k: int = Field(default=10, ge=1)
    relevance_threshold: float = 4.0
    context_events: int = Field(default=5, ge=1)  # streaming-value experiment


class ModelConfig(BaseModel):
    type: ModelType = "implicit_als"
    factors: int = Field(default=64, ge=1)
    regularization: float = Field(default=0.1, ge=0.0)
    alpha: float = Field(default=10.0, ge=0.0)  # implicit ALS confidence scale
    weighting: Literal["rating", "binary"] = "rating"  # implicit ALS confidence weights
    recency_half_life_days: float | None = None  # implicit ALS: decay confidence with age
    solver: Literal["cg", "exact"] = "cg"  # implicit ALS solver
    cg_steps: int = Field(default=3, ge=1)
    iterations: int = Field(default=15, ge=1)
    seed: int = 42
    popularity_half_life_days: float | None = None  # decay of the cold-start popularity ranking
    blend_weight: float = Field(default=0.0, ge=0.0)  # implicit ALS: trending-prior weight
    blend_half_life_days: float | None = None  # implicit ALS: decay of the trending prior
    backend: Literal["numpy", "torch"] = "numpy"  # ALS training: numpy (CPU) | torch (GPU)
    # SASRec (sequential transformer; ``factors`` is the hidden size). Trained with PyTorch
    # (requirements-gpu.txt), served on CPU through ONNX Runtime.
    max_len: int = Field(default=200, ge=2)  # most recent interactions per user
    num_blocks: int = Field(default=2, ge=1)
    num_heads: int = Field(default=1, ge=1)
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    batch_size: int = Field(default=128, ge=1)
    epochs: int = Field(default=200, ge=1)
    loss: Literal["ce", "sampled_ce"] = "ce"  # full softmax, or shared sampled negatives
    num_negatives: int = Field(default=1024, ge=1)  # sampled_ce only
    eval_every: int = Field(default=5, ge=1)  # epochs between validation checks (tuning)
    patience: int = Field(default=5, ge=1)  # validation checks without improvement
    device: str = "auto"  # auto | cpu | cuda
    # Ablation: "shuffled" permutes each user's training history, so the model never sees
    # the real order. If it scores as well, the order was not what helped.
    sequence_order: Literal["time", "shuffled"] = "time"


class GateConfig(BaseModel):
    metric: str = "ndcg_at_10"
    min_relative_improvement: float = 0.0  # challenger must beat champion by this ratio
    bootstrap_champion: bool = True  # first model ever becomes champion


class TuningConfig(BaseModel):
    grid: dict[str, list[float | int]] = Field(default_factory=dict)


class TrainConfig(BaseModel):
    dataset: str = "ml-latest-small"
    split: SplitConfig = Field(default_factory=SplitConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    baseline: ModelConfig = Field(default_factory=lambda: ModelConfig(type="popularity"))
    explicit: ModelConfig = Field(default_factory=lambda: ModelConfig(type="explicit_als"))
    sequential: ModelConfig | None = None  # SASRec, benchmarked when configured
    gate: GateConfig = Field(default_factory=GateConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with Path(path).open(encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh) or {})


def load_train_config(settings: Settings, dataset: str | None = None) -> TrainConfig:
    """Load ``configs/<dataset>.yaml``; fall back to defaults if the file does not exist."""
    path = settings.train_config_path(dataset)
    if path.exists():
        return TrainConfig.from_yaml(path)
    return TrainConfig(dataset=dataset or settings.dataset)
