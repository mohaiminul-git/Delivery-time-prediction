import argparse
import os
import sys

import pandas as pd

from src.config.configuration import FEATURE_ENGINEERED_TEST_DATA_PATH
from src.constants import TARGET_COLUMN
from src.exceptions import CustomException
from src.logger import logging
from src.pipeline.prediction_pipeline import PredictionPipeline

DEFAULT_OUTPUT_PATH = os.path.join("batch_prediction", "predictions.csv")


def run_batch_prediction(input_path: str = FEATURE_ENGINEERED_TEST_DATA_PATH, output_path: str = DEFAULT_OUTPUT_PATH):
    try:
        data = pd.read_csv(input_path)
        features = data.drop(columns=[TARGET_COLUMN], errors="ignore")

        predictions = PredictionPipeline().predict(features)

        result = features.copy()
        result["predicted_delivery_time"] = predictions

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        result.to_csv(output_path, index=False)

        logging.info(f"Batch prediction complete. Wrote {len(result)} rows to {output_path}")
        return output_path
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch delivery-time predictions on a CSV file.")
    parser.add_argument("--input", default=FEATURE_ENGINEERED_TEST_DATA_PATH, help="Path to the input CSV file.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Path to write predictions to.")
    args = parser.parse_args()

    output_path = run_batch_prediction(args.input, args.output)
    print(f"Predictions written to {output_path}")
