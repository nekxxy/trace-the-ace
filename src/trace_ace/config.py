"""Central, serializable configuration for training and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
ASSET_DIR = PROJECT_ROOT / "assets"
RUN_DIR = PROJECT_ROOT / "experiments" / "runs"
BUILD_DIR = PROJECT_ROOT / "submissions" / "builds"

SEED = 20_260_716
MODEL_VERSION = "cleanroom-v02"
BGE_REPOSITORY = "BAAI/bge-small-en-v1.5"
BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
BGE_DIMENSION = 384


@dataclass(frozen=True)
class ModelConfig:
    """Clean-room defaults used consistently by CV, final fit, and inference."""

    seed: int = SEED
    full_hash_features: int = 2**19
    role_hash_features: int = 2**18
    objective_hash_features: int = 2**16
    ngram_min: int = 1
    ngram_max: int = 2
    full_alpha: float = 3e-5
    role_alpha: float = 1e-4
    sparse_epochs: int = 4
    sparse_average: bool = True
    semantic_c: float = 0.1
    probability_floor: float = 1e-6
    full_weight: float = 0.25
    role_weight: float = 0.25
    semantic_weight: float = 0.50
    context_char_budget: int = 6_000
    bge_max_seq_length: int = 256
    bge_batch_size_cpu: int = 8
    bge_batch_size_gpu: int = 128

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SEMANTIC_PROTOCOLS: tuple[tuple[str, int, int], ...] = (
    ("semantic_k25_s0", 25, 0),
    ("semantic_k50_s0", 50, 0),
    ("semantic_k50_s1", 50, 1),
    ("semantic_k80_s0", 80, 0),
)
