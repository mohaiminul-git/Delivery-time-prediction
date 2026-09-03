import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.config.configuration import MODEL_TRAINER_PATH
from src.constants import DEFAULT_RANDOM_STATE
from src.exceptions import CustomException
from src.logger import logging
from src.utils import load_yaml, model_evaluate, save_obj

CONFIG_PATH = os.path.join("config", "config.yml")


@dataclass
class ModelTrainerConfig:
    model_trainer_path: str = MODEL_TRAINER_PATH


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1], train_arr[:, -1],
                test_arr[:, :-1], test_arr[:, -1],
            )

            random_state = load_yaml(CONFIG_PATH).get("model_trainer", {}).get("random_state", DEFAULT_RANDOM_STATE)

            models = {
                "Xgboost": XGBRegressor(random_state=random_state),
                "GradientBoosting": GradientBoostingRegressor(random_state=random_state),
                "DecisionTree": DecisionTreeRegressor(random_state=random_state),
                "RandomForest": RandomForestRegressor(random_state=random_state),
                "Svr": SVR(),
            }

            model_report: dict = model_evaluate(x_train, y_train, x_test, y_test, models)
            logging.info(f"Model evaluation report (R2 scores): {model_report}")

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name}, R2 score: {best_model_score}")

            save_obj(self.model_trainer_config.model_trainer_path, best_model)

            return best_model_name, best_model_score
        except Exception as e:
            raise CustomException(e, sys)
