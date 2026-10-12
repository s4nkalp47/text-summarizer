from textSummarizer.components.data_transformation import DataTransformation
from textSummarizer.config.configuration import ConfigurationManager


class DataTransformationPipeline:
    def __init__(self, config_manager: ConfigurationManager) -> None:
        self.config_manager = config_manager

    def main(self) -> None:
        config = self.config_manager.get_data_transformation_config()
        DataTransformation(config).convert()
