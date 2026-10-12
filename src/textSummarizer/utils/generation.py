import torch

from textSummarizer.entity import ModelEvaluationConfig, PredictionConfig
from textSummarizer.utils.common import clean_summary, input_prefix


def summarize_batch(model, tokenizer, dialogues: list[str], config: ModelEvaluationConfig | PredictionConfig) -> list[str]:
    """Summarize a batch of dialogues with the generation settings from params.yaml.

    Shared by evaluation and the API, so the served summaries match the measured ones exactly.
    """
    inputs = tokenizer(
        [input_prefix(tokenizer) + dialogue for dialogue in dialogues],
        max_length=config.max_input_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            num_beams=config.num_beams,
            length_penalty=config.length_penalty,
            max_new_tokens=config.max_new_tokens,
            min_length=config.min_length,
            no_repeat_ngram_size=config.no_repeat_ngram_size,
        )
    return [clean_summary(text) for text in tokenizer.batch_decode(output_ids, skip_special_tokens=True)]
