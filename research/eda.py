"""Exploratory analysis of the saved SAMSum dataset.

Prints split sizes, example dialogue/summary pairs and token-length percentiles,
proposes max_input_length / max_target_length, and saves a length histogram.

Run from the project root after `python main.py --stage ingestion`:
    python research/eda.py
"""
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write PNGs without needing a display
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer

from textSummarizer.config.configuration import ConfigurationManager

PERCENTILES = [50, 90, 95, 99]
NUM_EXAMPLES = 3
ROUND_TO = 64
MAX_INPUT_CAP = 1024  # Pegasus' position-embedding limit
HIST_PATH = Path("research/length_hist.png")

# Chart colors (single series per panel, light surface).
BAR_COLOR = "#2a78d6"
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def token_lengths(tokenizer, texts: list[str]) -> np.ndarray:
    # No truncation, so we see each text's true length (including the </s> token).
    return np.array([len(ids) for ids in tokenizer(texts, truncation=False)["input_ids"]])


def round_up(value: float, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def print_length_stats(name: str, lengths: np.ndarray) -> None:
    stats = "  ".join(f"p{p}={np.percentile(lengths, p):.0f}" for p in PERCENTILES)
    print(f"  {name:<9} {stats}  max={lengths.max()}  mean={lengths.mean():.0f}")


def plot_histograms(dialogue_lens: np.ndarray, summary_lens: np.ndarray, max_input: int, max_target: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=SURFACE)
    panels = [
        (axes[0], dialogue_lens, max_input, "Dialogue length (tokens)", "max_input_length"),
        (axes[1], summary_lens, max_target, "Summary length (tokens)", "max_target_length"),
    ]
    for ax, lengths, max_len, title, param in panels:
        ax.set_facecolor(SURFACE)
        x_max = max(lengths.max(), max_len)
        bin_width = max(1, math.ceil(x_max / 60))  # whole-token bins avoid aliasing on integer lengths
        ax.hist(lengths, bins=np.arange(0, x_max + bin_width, bin_width),
                color=BAR_COLOR, edgecolor=SURFACE, linewidth=0.8)
        ax.set_xlim(0, x_max * 1.05)
        p99 = np.percentile(lengths, 99)
        truncated = (lengths > max_len).mean() * 100
        ax.axvline(p99, color=TEXT_SECONDARY, linestyle=":", linewidth=1.5)
        ax.axvline(max_len, color=TEXT_PRIMARY, linestyle="--", linewidth=1.5)
        top = ax.get_ylim()[1]
        ax.text(p99, top * 0.95, f" p99 = {p99:.0f}", color=TEXT_SECONDARY, va="top", ha="right", fontsize=9)
        # Label to the right of the max line when there's room, otherwise to its left.
        room_on_right = max_len < 0.7 * x_max
        ax.text(max_len, top * 0.85, f" {param} = {max_len} \n {truncated:.1f}% truncated ",
                color=TEXT_PRIMARY, va="top", ha="left" if room_on_right else "right", fontsize=9)
        ax.set_title(title, color=TEXT_PRIMARY, loc="left", fontsize=11)
        ax.set_xlabel("Pegasus tokens", color=TEXT_SECONDARY)
        ax.set_ylabel("Training examples", color=TEXT_SECONDARY)
        ax.tick_params(colors=TEXT_SECONDARY)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    fig.suptitle("SAMSum train split: token lengths", color=TEXT_PRIMARY, x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(HIST_PATH, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    # smoke=False: always measure with the real Pegasus tokenizer, not the smoke model's.
    config_manager = ConfigurationManager(smoke=False)
    data_path = config_manager.get_data_ingestion_config().data_path
    dataset = load_from_disk(data_path)

    print("\n=== Split sizes ===")
    for split, rows in dataset.items():
        print(f"  {split:<11} {len(rows):>6}")

    train = dataset["train"]
    print(f"\n=== {NUM_EXAMPLES} example pairs (train) ===")
    for example in train.select(range(NUM_EXAMPLES)):
        print(f"\n--- id {example['id']} ---")
        print(example["dialogue"])
        print(f">> SUMMARY: {example['summary']}")

    print(f"\nLoading tokenizer: {config_manager.model_ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(config_manager.model_ckpt)
    # datasets >= 4 returns a lazy Column for train["col"]; the tokenizer needs a real list.
    dialogue_lens = token_lengths(tokenizer, list(train["dialogue"]))
    summary_lens = token_lengths(tokenizer, list(train["summary"]))

    print("\n=== Token lengths (train) ===")
    print_length_stats("dialogue", dialogue_lens)
    print_length_stats("summary", summary_lens)

    max_input = min(round_up(np.percentile(dialogue_lens, 99), ROUND_TO), MAX_INPUT_CAP)
    max_target = round_up(np.percentile(summary_lens, 99), ROUND_TO)
    print("\n=== Proposed params.yaml values (p99 rounded up to a multiple of 64) ===")
    print(f"  max_input_length:  {max_input}  ({(dialogue_lens > max_input).mean():.2%} of dialogues truncated)")
    print(f"  max_target_length: {max_target}  ({(summary_lens > max_target).mean():.2%} of summaries truncated)")

    plot_histograms(dialogue_lens, summary_lens, max_input, max_target)
    print(f"\nSaved histogram to {HIST_PATH}")


if __name__ == "__main__":
    main()
