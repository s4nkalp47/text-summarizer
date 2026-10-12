from textSummarizer.components.model_trainer import ModelTrainer
from textSummarizer.config.configuration import ConfigurationManager


class ModelTrainerPipeline:
    def __init__(self, config_manager: ConfigurationManager) -> None:
        self.config_manager = config_manager

    def main(self) -> None:
        config = self.config_manager.get_model_trainer_config()
        ModelTrainer(config).train()
