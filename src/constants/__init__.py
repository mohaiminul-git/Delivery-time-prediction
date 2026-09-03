import os
from datetime import datetime


def get_current_time_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H-%M-%S")


CURRENT_TIME_STAMP = get_current_time_stamp()

ROOT_DIR = os.getcwd()
DATA_DIR = "data"
DATA_FILE_NAME = "finalTrain.csv"

ARTIFACT_DIR = "artifacts"
DATA_INGESTION_DIR = "data_ingestion"
RAW_DATA_DIR = "raw_data"
INGESTED_DATA_DIR = "ingested_data"
RAW_DATA_FILE_NAME = "raw.csv"
TRAIN_DATA_FILE_NAME = "train.csv"
TEST_DATA_FILE_NAME = "test.csv"

# data transformation
DATA_TRANSFORMATION_ARTIFACT_DIR = "data_transformation"
DATA_PROCESSOR_DIR = "processor"
DATA_PROCESSOR_FILE_NAME = "processor.pkl"
FEATURE_ENGINEERING_FILE_NAME = "feature_engineering.pkl"
TRANSFORMED_DATA_DIR = "transformed_data"
TRANSFORMED_TRAIN_DATA_FILE_NAME = "train.csv"
TRANSFORMED_TEST_DATA_FILE_NAME = "test.csv"

# model trainer
MODEL_TRAINER_ARTIFACT_DIR = "model_trainer"
MODEL_TRAINER_FILE_NAME = "trained_model.pkl"

# training config (overridable via config/config.yml)
DEFAULT_TEST_SIZE = 0.30
DEFAULT_RANDOM_STATE = 42

TARGET_COLUMN = "Time_taken (min)"
