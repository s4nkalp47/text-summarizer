"""Upload the fine-tuned model and tokenizer to a Hugging Face Hub repo.

Model and tokenizer files go to the repo root, so one id loads both:
    AutoModelForSeq2SeqLM.from_pretrained("<user>/<repo>"), AutoTokenizer.from_pretrained("<user>/<repo>")

Log in first (`hf auth login`), then run from the project root:
    python research/push_to_hub.py <your-username>/pegasus-samsum            # private repo (default)
    python research/push_to_hub.py <your-username>/pegasus-samsum --public   # anyone can download it
"""
import argparse
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

from textSummarizer.config.configuration import ConfigurationManager

MODEL_CARD = """---
library_name: transformers
pipeline_tag: summarization
base_model: {base_model}
datasets:
- knkarthick/samsum
language:
- en
license: other
---

# {repo_id}

`{base_model}` fine-tuned for 1 epoch on [SAMSum](https://huggingface.co/datasets/knkarthick/samsum)
to summarize messenger-style dialogues.

{metrics}

**Generation settings used for these scores:** {generation}

## Limitations
Summaries are fluent but can state false facts, mostly negation ("will buy" for "needn't buy") and
who-did-what confusions. Check important details against the chat.

## License
The base model is Apache-2.0. SAMSum is licensed **CC BY-NC-ND 4.0** (non-commercial), so this model
must not be used commercially.
"""


def metrics_section(metrics_path: Path) -> str:
    if not metrics_path.exists():
        return ""
    row = pd.read_csv(metrics_path).iloc[0]
    scores = " | ".join(f"{row[key]:.4f}" for key in ["rouge1", "rouge2", "rougeL", "rougeLsum"])
    return (
        f"## Results ({int(row['num_samples'])} SAMSum test dialogues)\n\n"
        "| ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |\n|---|---|---|---|\n"
        f"| {scores} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("repo_id", help="Hub repo to create or update, e.g. <your-username>/pegasus-samsum")
    parser.add_argument("--public", action="store_true", help="Create the repo as public (default: private).")
    args = parser.parse_args()

    config_manager = ConfigurationManager(smoke=False)
    trainer_config = config_manager.get_model_trainer_config()
    eval_config = config_manager.get_model_evaluation_config()
    generation = config_manager.params.generation
    for folder in (trainer_config.model_path, trainer_config.tokenizer_path):
        if not (folder.exists() and any(folder.iterdir())):
            raise SystemExit(f"{folder} is missing or empty: train the model (or copy it from Colab) first.")

    api = HfApi()
    url = api.create_repo(args.repo_id, private=not args.public, exist_ok=True)
    print(f"Uploading to {url} ({'public' if args.public else 'private'}). The 2.3 GB model may take a while.")
    api.upload_folder(repo_id=args.repo_id, folder_path=trainer_config.model_path, commit_message="Upload fine-tuned model")
    api.upload_folder(repo_id=args.repo_id, folder_path=trainer_config.tokenizer_path, commit_message="Upload tokenizer")

    card = MODEL_CARD.format(
        repo_id=args.repo_id,
        base_model=config_manager.config.base_model_ckpt,
        metrics=metrics_section(eval_config.metrics_path),
        generation=", ".join(f"{key}={value}" for key, value in generation.items()),
    )
    api.upload_file(repo_id=args.repo_id, path_or_fileobj=card.encode(), path_in_repo="README.md", commit_message="Add model card")
    print(f"Done: {url}\nServe it with MODEL_ID={args.repo_id}")


if __name__ == "__main__":
    main()
