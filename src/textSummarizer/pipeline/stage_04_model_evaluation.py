from textSummarizer.components.model_evaluation import ModelEvaluation
from textSummarizer.config.configuration import ConfigurationManager


class ModelEvaluationPipeline:
    def __init__(self, config_manager: ConfigurationManager) -> None:
        self.config_manager = config_manager

    def main(self) -> None:
        config = self.config_manager.get_model_evaluation_config()
        ModelEvaluation(config).evaluate()
