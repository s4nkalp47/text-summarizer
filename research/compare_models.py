"""Compare the pretrained base checkpoint with the fine-tuned model on the same test dialogues.

Writes ROUGE scores for both models to comparison.csv (path from config.yaml).
Run from the project root after training:
    python research/compare_models.py            # full: Pegasus base vs fine-tuned (use a GPU)
    python research/compare_models.py --smoke    # quick check: t5-small base vs smoke model
"""
import argparse
import gc

import pandas as pd
import torch

from textSummarizer.components.model_evaluation import ModelEvaluation, evaluate_model
from textSummarizer.config.configuration import ConfigurationManager


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", help="Compare the smoke models instead of Pegasus.")
    args = parser.parse_args()

    config = ConfigurationManager(smoke=args.smoke).get_model_evaluation_config()
    dialogues, references = ModelEvaluation(config).load_test_samples()
    models = {
        "base": (config.base_model_ckpt, config.base_model_ckpt),
        "fine-tuned": (str(config.model_path), str(config.tokenizer_path)),
    }

    rows, predictions = [], {}
    for label, (model_path, tokenizer_path) in models.items():
        scores, predictions[label] = evaluate_model(model_path, tokenizer_path, dialogues, references, config)
        rows.append({"model": label, "checkpoint": model_path, "num_samples": len(dialogues), **scores})
        gc.collect()  # free the first model before loading the second
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    comparison = pd.DataFrame(rows)
    comparison.to_csv(config.comparison_path, index=False)
    print(f"\n=== ROUGE on {len(dialogues)} test dialogues ===")
    print(comparison.drop(columns="checkpoint").to_string(index=False))

    print("\n=== First 3 test dialogues: base vs fine-tuned ===")
    for i in range(min(3, len(dialogues))):
        print(f"\n--- {i + 1} ---")
        print(f"REFERENCE:  {references[i]}")
        print(f"BASE:       {predictions['base'][i]}")
        print(f"FINE-TUNED: {predictions['fine-tuned'][i]}")
    print(f"\nSaved {config.comparison_path}")


if __name__ == "__main__":
    main()
