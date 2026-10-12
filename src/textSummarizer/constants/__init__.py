from pathlib import Path

# Relative to the project root: run commands from there.
CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")

# Written next to the tokenized dataset so training can check it matches its checkpoint.
TOKENIZER_NAME_FILE = "tokenizer_name.txt"

# T5 checkpoints expect the task prefix on every input (training and inference).
T5_INPUT_PREFIX = "summarize: "

# Environment variable that overrides prediction.model_path/tokenizer_path, e.g. with a Hub id in Docker.
MODEL_ID_ENV_VAR = "MODEL_ID"
