import logging
import os
from datetime import datetime

LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

CURRENT_TIME_STAMP = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
LOG_FILE_NAME = f"log_{CURRENT_TIME_STAMP}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

# Create the file up front: logging.basicConfig() below is a no-op if a root
# handler is already configured (e.g. by pytest), which would otherwise leave
# no file on disk even though the module loaded successfully.
open(LOG_FILE_PATH, "a").close()

logging.basicConfig(
    filename=LOG_FILE_PATH,
    filemode="w",
    format="%(asctime)s :: %(levelname)s :: %(message)s",
    level=logging.INFO,
)
