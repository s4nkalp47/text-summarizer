from dataclasses import replace
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

from textSummarizer.components.data_transformation import DataTransformation
from textSummarizer.entity import DataTransformationConfig

# Small limits so the long example below is guaranteed to be truncated.
MAX_INPUT_LENGTH = 32
MAX_TARGET_LENGTH = 8

ROWS = {
    "id": ["1", "2", "3"],
    "dialogue": [
        "Amanda: I baked cookies. Do you want some?\nJerry: Sure!",
        "Tom: Are we meeting at 5?\nAnna: Yes, see you there.",
        "Kim: " + "I really need to tell you about my long day at work. " * 20,
    ],
    "summary": [
        "Amanda baked cookies for Jerry.",
        "Tom and Anna are meeting at 5.",
        "Kim had a very long and tiring day at work and wants to talk about all of it in detail.",
    ],
}


@pytest.fixture(scope="module")
def transformation(tmp_path_factory: pytest.TempPathFactory) -> DataTransformation:
    root = tmp_path_factory.mktemp("data_transformation")
    config = DataTransformationConfig(
        root_dir=root,
        data_path=Path("unused"),
        output_path=root / "tokenized",
        tokenizer_name="t5-small",  # tokenizer only: a small download, no model weights
        max_input_length=MAX_INPUT_LENGTH,
        max_target_length=MAX_TARGET_LENGTH,
        num_samples=None,
    )
    return DataTransformation(config)


@pytest.fixture(scope="module")
def tokenized(transformation: DataTransformation) -> Dataset:
    return Dataset.from_dict(ROWS).map(
        transformation.tokenize_batch, batched=True, remove_columns=list(ROWS)
    )


def test_output_has_only_model_columns(tokenized: Dataset) -> None:
    assert set(tokenized.column_names) == {"input_ids", "attention_mask", "labels"}


def test_no_sequence_exceeds_max_lengths(tokenized: Dataset) -> None:
    assert all(len(ids) <= MAX_INPUT_LENGTH for ids in tokenized["input_ids"])
    assert all(len(ids) <= MAX_TARGET_LENGTH for ids in tokenized["labels"])
    # The long third row must actually have been truncated to the limits.
    assert len(tokenized[2]["input_ids"]) == MAX_INPUT_LENGTH
    assert len(tokenized[2]["labels"]) == MAX_TARGET_LENGTH


def test_t5_prefix_added_to_inputs_only(transformation: DataTransformation, tokenized: Dataset) -> None:
    decoded_input = transformation.tokenizer.decode(tokenized[0]["input_ids"], skip_special_tokens=True)
    decoded_label = transformation.tokenizer.decode(tokenized[0]["labels"], skip_special_tokens=True)

    assert decoded_input.startswith("summarize: Amanda")
    assert decoded_label == "Amanda baked cookies for Jerry."


def test_transform_keeps_num_samples_rows_per_split(transformation: DataTransformation) -> None:
    dataset = DatasetDict({"train": Dataset.from_dict(ROWS), "test": Dataset.from_dict(ROWS)})
    limited = DataTransformation(replace(transformation.config, num_samples=2))

    result = limited.transform(dataset)

    assert {split: len(rows) for split, rows in result.items()} == {"train": 2, "test": 2}
