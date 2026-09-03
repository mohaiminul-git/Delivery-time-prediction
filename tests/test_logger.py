import os

from src.logger import LOG_FILE_PATH


def test_log_file_is_created():
    assert os.path.exists(LOG_FILE_PATH)


def test_log_file_lives_in_logs_directory():
    assert os.path.basename(os.path.dirname(LOG_FILE_PATH)) == "logs"
