import re

import evaluate
import pandas as pd
import torch
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from textSummarizer.entity import ModelEvaluationConfig
from textSummarizer.logging import logger
from textSummarizer.utils.generation import summarize_batch

ROUGE_KEYS = ["rouge1", "rouge2", "rougeL", "rougeLsum"]


def generate_summaries(model, tokenizer, dialogues: list[str], config: ModelEvaluationConfig) -> list[str]:
    """Summarize dialogues in batches of evaluation.batch_size."""
    summaries = []
    for start in range(0, len(dialogues), config.batch_size):
        summaries += summarize_batch(model, tokenizer, dialogues[start : start + config.batch_size], config)
        logger.info(f"Summarized {min(start + config.batch_size, len(dialogues))}/{len(dialogues)} dialogues")
    return summaries


def split_sentences(text: str) -> str:
    """One sentence per line: rougeLsum scores sentence by sentence, using newlines as sentence breaks."""
    return "\n".join(re.split(r"(?<=[.!?])\s+", text.strip()))


def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    rouge = evaluate.load("rouge")
    scores = rouge.compute(
        predictions=[split_sentences(p) for p in predictions],
        references=[split_sentences(r) for r in references],
        rouge_types=ROUGE_KEYS,
        use_stemmer=True,  # count "bake" and "baked" as a match, as in the ROUGE papers
    )
    return {key: round(float(scores[key]), 4) for key in ROUGE_KEYS}


def evaluate_model(
    model_path: str, tokenizer_path: str, dialogues: list[str], references: list[str], config: ModelEvaluationConfig
) -> tuple[dict[str, float], list[str]]:
    """Load a model (local folder or Hub id), summarize the dialogues, and score them against the references."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Evaluating {model_path} on {len(dialogues)} test dialogues ({device})")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device).eval()
    predictions = generate_summaries(model, tokenizer, dialogues, config)
    scores = compute_rouge(predictions, references)
    logger.info(f"ROUGE for {model_path}: {scores}")
    return scores, predictions


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig) -> None:
        self.config = config

    def load_test_samples(self) -> tuple[list[str], list[str]]:
        test = load_from_disk(self.config.data_path)["test"]
        test = test.select(range(min(self.config.num_samples, len(test))))
        return list(test["dialogue"]), list(test["summary"])

    def evaluate(self) -> None:
        dialogues, references = self.load_test_samples()
        scores, predictions = evaluate_model(
            str(self.config.model_path), str(self.config.tokenizer_path), dialogues, references, self.config
        )

        pd.DataFrame([{"model": str(self.config.model_path), "num_samples": len(dialogues), **scores}]).to_csv(
            self.config.metrics_path, index=False
        )
        n = self.config.num_saved_samples
        pd.DataFrame(
            {"dialogue": dialogues[:n], "reference": references[:n], "prediction": predictions[:n]}
        ).to_csv(self.config.samples_path, index=False)
        logger.info(f"Saved metrics to {self.config.metrics_path} and {n} samples to {self.config.samples_path}")
