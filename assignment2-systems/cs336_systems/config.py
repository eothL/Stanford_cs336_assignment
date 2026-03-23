from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

@dataclass(slots=True)
class ModelConfig:
    vocab_size:int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int 
    rope_theta: float = 10_000.0

@dataclass(slots=True)
class TrainingConfig:
    model: ModelConfig
    batch_size: int = 8
    max_steps: int = 1_000
    device: str = "cpu"
    train_dataset: str | Path | None = None
    val_dataset: str | Path | None = None
    learning_rate: float = 3e-4
    seed: int = 93
