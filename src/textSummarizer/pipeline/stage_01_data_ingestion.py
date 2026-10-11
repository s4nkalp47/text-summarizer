from textSummarizer.components.data_ingestion import DataIngestion
from textSummarizer.config.configuration import ConfigurationManager


class DataIngestionPipeline:
    def __init__(self, config_manager: ConfigurationManager) -> None:
        self.config_manager = config_manager

    def main(self) -> None:
        config = self.config_manager.get_data_ingestion_config()
        DataIngestion(config).download_data()
