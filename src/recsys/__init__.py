"""MLOps recommendation pipeline: data, features, models, registry, serving, monitoring."""

import os

# MLflow >= 3.x prints an "agent hint" banner on import and a progress bar for every
# artifact download (model loads, promotions); both are noise in CLI and service logs.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

__version__ = "0.1.0"
