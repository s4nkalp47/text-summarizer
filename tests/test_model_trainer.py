from pathlib import Path

import pytest

from textSummarizer.components.model_trainer import ModelTrainer
from textSummarizer.constants import TOKENIZER_NAME_FILE
from textSummarizer.entity import ModelTrainerConfig


def make_trainer(data_path: Path, model_ckpt: str) -> ModelTrainer:
    config = ModelTrainerConfig(
        root_dir=data_path.parent,
        data_path=data_path,
        model_path=data_path.parent / "model",
        tokenizer_path=data_path.parent / "tokenizer",
        model_ckpt=model_ckpt,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=5.0e-5,
        warmup_steps=0,
        weight_decay=0.0,
        logging_steps=1,
        fp16=False,
        optim="adamw_torch",
        gradient_checkpointing=False,
        max_steps=1,
    )
    return ModelTrainer(config)


def test_data_check_passes_when_tokenizer_matches(tmp_path: Path) -> None:
    (tmp_path / TOKENIZER_NAME_FILE).write_text("t5-small")

    make_trainer(tmp_path, "t5-small").check_data_matches_checkpoint()


def test_data_check_rejects_data_from_another_tokenizer(tmp_path: Path) -> None:
    (tmp_path / TOKENIZER_NAME_FILE).write_text("t5-small")

    with pytest.raises(ValueError, match="tokenized with 't5-small'"):
        make_trainer(tmp_path, "google/pegasus-cnn_dailymail").check_data_matches_checkpoint()


def test_data_check_rejects_data_without_tokenizer_record(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="'unknown'"):
        make_trainer(tmp_path, "t5-small").check_data_matches_checkpoint()
