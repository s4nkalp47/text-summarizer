"""Try several decoding settings on VALIDATION dialogues to pick the generation params.

Tuning on the test split would make the final test score optimistic, so this script never touches it.
Writes to the evaluation stage's folder:
  decoding_sweep.csv          ROUGE, average summary length and runtime per setting
  decoding_sweep_samples.csv  each setting's summary side by side, for labeling errors by hand

Run from the project root after training:
    python research/decoding_sweep.py            # fine-tuned Pegasus (slow on CPU: ~4 min per setting)
    python research/decoding_sweep.py --smoke    # quick plumbing check with the smoke model
"""
import argparse
import time
from dataclasses import replace

import pandas as pd
import torch
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from textSummarizer.components.model_evaluation import compute_rouge, generate_summaries
from textSummarizer.config.configuration import ConfigurationManager

# Each setting overrides params.yaml's `generation` values; "current" changes nothing.
SETTINGS = {
    "current": {},
    "no_repeat_3": {"no_repeat_ngram_size": 3},
    "min_len_10": {"min_length": 10},
    "min_len_15": {"min_length": 15},
    "len_penalty_1.2": {"length_penalty": 1.2},
    "beams_4": {"num_beams": 4},
}
NUM_LABEL_SAMPLES = 30  # rows in decoding_sweep_samples.csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", help="Use the smoke model and 10 dialogues.")
    args = parser.parse_args()

    base_config = ConfigurationManager(smoke=args.smoke).get_model_evaluation_config()
    validation = load_from_disk(base_config.data_path)["validation"]
    validation = validation.select(range(min(base_config.num_samples, len(validation))))
    dialogues, references = list(validation["dialogue"]), list(validation["summary"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base_config.tokenizer_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_config.model_path).to(device).eval()
    print(f"Sweeping {len(SETTINGS)} settings on {len(dialogues)} validation dialogues ({device})")

    rows = []
    samples = pd.DataFrame(
        {"dialogue": dialogues[:NUM_LABEL_SAMPLES], "reference": references[:NUM_LABEL_SAMPLES]}
    )
    for name, overrides in SETTINGS.items():
        config = replace(base_config, **overrides)
        start = time.time()
        predictions = generate_summaries(model, tokenizer, dialogues, config)
        seconds = time.time() - start
        scores = compute_rouge(predictions, references)
        avg_words = sum(len(p.split()) for p in predictions) / len(predictions)
        rows.append({"setting": name, **overrides, **scores, "avg_words": round(avg_words, 1), "seconds": round(seconds)})
        samples[name] = predictions[:NUM_LABEL_SAMPLES]
        print(f"{name:<16} rouge1={scores['rouge1']:.4f} rouge2={scores['rouge2']:.4f} "
              f"avg_words={avg_words:.1f} time={seconds:.0f}s")

    reference_words = sum(len(r.split()) for r in references) / len(references)
    results = pd.DataFrame(rows)
    results_path = base_config.root_dir / "decoding_sweep.csv"
    samples_path = base_config.root_dir / "decoding_sweep_samples.csv"
    results.to_csv(results_path, index=False)
    samples.to_csv(samples_path, index=False)

    print(f"\n=== Validation ROUGE by decoding setting (references average {reference_words:.1f} words) ===")
    print(results.fillna("").to_string(index=False))
    print(f"\nSaved {results_path} and {samples_path}")


if __name__ == "__main__":
    main()
