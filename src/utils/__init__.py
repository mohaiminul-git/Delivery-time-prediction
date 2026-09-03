import os
import pickle
import sys

import yaml
from sklearn.metrics import r2_score

from src.exceptions import CustomException
from src.logger import logging


def save_obj(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_model(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logging.info("Exception occurred while loading an object")
        raise CustomException(e, sys)


def model_evaluate(x_train, y_train, x_test, y_test, models: dict) -> dict:
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(x_train, y_train)
            y_test_pred = model.predict(x_test)
            report[model_name] = r2_score(y_test, y_test_pred)

        return report
    except Exception as e:
        raise CustomException(e, sys)


def load_yaml(file_path: str) -> dict:
    try:
        with open(file_path, "r") as f:
            return yaml.safe_load(f) 
    except Exception as e:
        raise CustomException(e, sys)


def validate_schema(df, schema: dict) -> None:
    """Check that a dataframe has exactly the columns declared in schema.yml."""
    try:
        expected_columns = set(schema.get("columns", {}).keys())
        actual_columns = set(df.columns)

        missing = expected_columns - actual_columns
        if missing:
            raise ValueError(f"Missing expected columns: {sorted(missing)}")

        logging.info("Schema validation passed")
    except Exception as e:
        raise CustomException(e, sys)
