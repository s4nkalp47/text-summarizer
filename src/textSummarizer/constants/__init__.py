from pathlib import Path

# Relative to the project root: run commands from there.
CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")

# T5 checkpoints expect the task prefix on every input (training and inference).
T5_INPUT_PREFIX = "summarize: "
