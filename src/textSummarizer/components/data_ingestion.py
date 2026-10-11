from datasets import load_dataset

from textSummarizer.entity import DataIngestionConfig
from textSummarizer.logging import logger


class DataIngestion:
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def download_data(self) -> None:
        """Download the dataset from the Hugging Face Hub and save it to disk, unless it's already saved."""
        # save_to_disk writes dataset_dict.json last, so its presence means a complete save.
        if (self.config.data_path / "dataset_dict.json").exists():
            logger.info(f"Dataset already saved at {self.config.data_path}, skipping download")
            return

        logger.info(f"Downloading dataset '{self.config.dataset_name}' from the Hugging Face Hub")
        dataset = load_dataset(self.config.dataset_name)
        dataset.save_to_disk(self.config.data_path)
        split_sizes = {split: len(rows) for split, rows in dataset.items()}
        logger.info(f"Saved dataset to {self.config.data_path} with splits {split_sizes}")
