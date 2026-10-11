from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    dataset_name: str
    data_path: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    output_path: Path
    tokenizer_name: str
    max_input_length: int
    max_target_length: int
    num_samples: Optional[int]  # rows kept per split; None = all


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    model_path: Path
    tokenizer_path: Path
    model_ckpt: str
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_steps: int
    weight_decay: float
    logging_steps: int
    fp16: bool
    optim: str
    gradient_checkpointing: bool
    max_steps: int  # -1 = train for num_train_epochs (Transformers' convention)


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    model_path: Path
    tokenizer_path: Path
    metrics_path: Path
    samples_path: Path
    num_samples: int
    batch_size: int
    max_input_length: int
    num_beams: int
    length_penalty: float
    max_new_tokens: int


@dataclass(frozen=True)
class PredictionConfig:
    model_path: str  # local folder or Hugging Face Hub id
    tokenizer_path: str
    max_input_length: int
    num_beams: int
    length_penalty: float
    max_new_tokens: int
