import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.configuration import (
    INGESTED_RAW_DATA_PATH,
    RAW_DATA_PATH,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
)
from src.constants import DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE
from src.exceptions import CustomException
from src.logger import logging
from src.utils import load_yaml, validate_schema

CONFIG_PATH = os.path.join("config", "config.yml")
SCHEMA_PATH = "schema.yml"


@dataclass
class DataIngestionConfig:
    train_data_path: str = TRAIN_DATA_PATH
    test_data_path: str = TEST_DATA_PATH
    raw_data_path: str = INGESTED_RAW_DATA_PATH


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            config = load_yaml(CONFIG_PATH).get("data_ingestion", {})
            test_size = config.get("test_size", DEFAULT_TEST_SIZE)
            random_state = config.get("random_state", DEFAULT_RANDOM_STATE)

            data = pd.read_csv(RAW_DATA_PATH)

            schema = load_yaml(SCHEMA_PATH)
            if schema:
                validate_schema(data, schema)

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)
            data.to_csv(self.data_ingestion_config.raw_data_path, index=False)

            train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)
            train_data.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)

            os.makedirs(os.path.dirname(self.data_ingestion_config.test_data_path), exist_ok=True)
            test_data.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed")

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
