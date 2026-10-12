import logging
import os
import sys

LOG_DIR = "logs"
LOG_FILEPATH = os.path.join(LOG_DIR, "running_logs.log")
LOG_FORMAT = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("textSummarizer")
logger.setLevel(logging.INFO)
# Libraries like `evaluate` configure the root logger; without this, every line would print twice.
logger.propagate = False

# Guard so handlers aren't added twice if this module is re-imported.
if not logger.handlers:
    formatter = logging.Formatter(LOG_FORMAT)
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILEPATH, encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
