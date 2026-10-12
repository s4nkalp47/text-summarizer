from datasets import DatasetDict, load_from_disk
from transformers import AutoTokenizer

from textSummarizer.constants import TOKENIZER_NAME_FILE
from textSummarizer.entity import DataTransformationConfig
from textSummarizer.logging import logger
from textSummarizer.utils.common import input_prefix


class DataTransformation:
    def __init__(self, config: DataTransformationConfig) -> None:
        self.config = config
        logger.info(f"Loading tokenizer: {config.tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.input_prefix = input_prefix(self.tokenizer)

    def tokenize_batch(self, batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        """Tokenize dialogues as model inputs and summaries as labels (used with Dataset.map(batched=True))."""
        inputs = [self.input_prefix + dialogue for dialogue in batch["dialogue"]]
        model_inputs = self.tokenizer(inputs, max_length=self.config.max_input_length, truncation=True)
        targets = self.tokenizer(
            text_target=batch["summary"], max_length=self.config.max_target_length, truncation=True
        )
        model_inputs["labels"] = targets["input_ids"]
        return model_inputs

    def transform(self, dataset: DatasetDict) -> DatasetDict:
        if self.config.num_samples is not None:
            dataset = DatasetDict(
                {split: rows.select(range(min(self.config.num_samples, len(rows)))) for split, rows in dataset.items()}
            )
        return dataset.map(
            self.tokenize_batch,
            batched=True,
            remove_columns=dataset["train"].column_names,  # keep only input_ids, attention_mask, labels
        )

    def convert(self) -> None:
        dataset = load_from_disk(self.config.data_path)
        tokenized = self.transform(dataset)
        tokenized.save_to_disk(self.config.output_path)
        (self.config.output_path / TOKENIZER_NAME_FILE).write_text(self.config.tokenizer_name, encoding="utf-8")
        split_sizes = {split: len(rows) for split, rows in tokenized.items()}
        logger.info(
            f"Saved tokenized dataset to {self.config.output_path} with splits {split_sizes} "
            f"and columns {tokenized['train'].column_names}"
        )
