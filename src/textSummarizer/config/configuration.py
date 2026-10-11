from pathlib import Path
from typing import Optional

from textSummarizer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from textSummarizer.entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelEvaluationConfig,
    ModelTrainerConfig,
    PredictionConfig,
)
from textSummarizer.logging import logger
from textSummarizer.utils.common import create_directories, read_yaml


class ConfigurationManager:
    """Reads config.yaml and params.yaml and builds each stage's config dataclass.

    smoke=None follows smoke_test.enabled in params.yaml; True/False overrides it.
    """

    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
        smoke: Optional[bool] = None,
    ) -> None:
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.smoke = self.params.smoke_test.enabled if smoke is None else smoke
        self.model_ckpt = (
            self.params.smoke_test.model_ckpt if self.smoke else self.config.base_model_ckpt
        )
        create_directories([self.config.artifacts_root])
        logger.info(f"Smoke mode: {'ON' if self.smoke else 'OFF'} | model checkpoint: {self.model_ckpt}")

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            dataset_name=config.dataset_name,
            data_path=Path(config.data_path),
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        tokenization = self.params.tokenization
        create_directories([config.root_dir])
        return DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            output_path=Path(config.output_path),
            tokenizer_name=self.model_ckpt,
            max_input_length=tokenization.max_input_length,
            max_target_length=tokenization.max_target_length,
            num_samples=self.params.smoke_test.num_samples if self.smoke else None,
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        training = self.params.training
        create_directories([config.root_dir])
        return ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            model_path=Path(config.model_path),
            tokenizer_path=Path(config.tokenizer_path),
            model_ckpt=self.model_ckpt,
            num_train_epochs=training.num_train_epochs,
            per_device_train_batch_size=training.per_device_train_batch_size,
            gradient_accumulation_steps=training.gradient_accumulation_steps,
            learning_rate=training.learning_rate,
            warmup_steps=training.warmup_steps,
            weight_decay=training.weight_decay,
            logging_steps=training.logging_steps,
            fp16=training.fp16,
            optim=training.optim,
            gradient_checkpointing=training.gradient_checkpointing,
            max_steps=self.params.smoke_test.max_steps if self.smoke else -1,
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        evaluation = self.params.evaluation
        generation = self.params.generation
        create_directories([config.root_dir])
        return ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            model_path=Path(config.model_path),
            tokenizer_path=Path(config.tokenizer_path),
            metrics_path=Path(config.metrics_path),
            samples_path=Path(config.samples_path),
            num_samples=self.params.smoke_test.num_samples if self.smoke else evaluation.num_samples,
            batch_size=evaluation.batch_size,
            max_input_length=self.params.tokenization.max_input_length,
            num_beams=generation.num_beams,
            length_penalty=generation.length_penalty,
            max_new_tokens=generation.max_new_tokens,
        )

    def get_prediction_config(self) -> PredictionConfig:
        config = self.config.prediction
        generation = self.params.generation
        return PredictionConfig(
            model_path=str(config.model_path),
            tokenizer_path=str(config.tokenizer_path),
            max_input_length=self.params.tokenization.max_input_length,
            num_beams=generation.num_beams,
            length_penalty=generation.length_penalty,
            max_new_tokens=generation.max_new_tokens,
        )
