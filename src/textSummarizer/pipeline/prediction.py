import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from textSummarizer.entity import PredictionConfig
from textSummarizer.logging import logger
from textSummarizer.utils.generation import summarize_batch


class PredictionPipeline:
    """Loads the model once, then summarizes one dialogue per call."""

    def __init__(self, config: PredictionConfig) -> None:
        self.config = config
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading model {config.model_path} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_path).to(device).eval()

    def summarize(self, dialogue: str) -> str:
        return summarize_batch(self.model, self.tokenizer, [dialogue], self.config)[0]
