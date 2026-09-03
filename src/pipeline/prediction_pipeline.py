import sys

import pandas as pd

from src.config.configuration import MODEL_TRAINER_PATH, PREPROCESSING_OBJ_PATH
from src.exceptions import CustomException
from src.logger import logging
from src.utils import load_model


class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        try:
            preprocessor = load_model(PREPROCESSING_OBJ_PATH)
            model = load_model(MODEL_TRAINER_PATH)

            data_scaled = preprocessor.transform(features)
            return model.predict(data_scaled)
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        Type_of_order: str,
        Type_of_vehicle: str,
        Festival: str,
        City: str,
        Delivery_city: str,
        Road_traffic_density: str,
        Weather_conditions: str,
        Delivery_person_Age: float,
        Delivery_person_Ratings: float,
        Vehicle_condition: float,
        multiple_deliveries: float,
        Time_Orderd_hour: float,
        distance: float,
    ):
        self.Type_of_order = Type_of_order
        self.Type_of_vehicle = Type_of_vehicle
        self.Festival = Festival
        self.City = City
        self.Delivery_city = Delivery_city
        self.Road_traffic_density = Road_traffic_density
        self.Weather_conditions = Weather_conditions
        self.Delivery_person_Age = Delivery_person_Age
        self.Delivery_person_Ratings = Delivery_person_Ratings
        self.Vehicle_condition = Vehicle_condition
        self.multiple_deliveries = multiple_deliveries
        self.Time_Orderd_hour = Time_Orderd_hour
        self.distance = distance

    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "Type_of_order": [self.Type_of_order],
                "Type_of_vehicle": [self.Type_of_vehicle],
                "Festival": [self.Festival],
                "City": [self.City],
                "Delivery_city": [self.Delivery_city],
                "Road_traffic_density": [self.Road_traffic_density],
                "Weather_conditions": [self.Weather_conditions],
                "Delivery_person_Age": [self.Delivery_person_Age],
                "Delivery_person_Ratings": [self.Delivery_person_Ratings],
                "Vehicle_condition": [self.Vehicle_condition],
                "multiple_deliveries": [self.multiple_deliveries],
                "Time_Orderd_hour": [self.Time_Orderd_hour],
                "distance": [self.distance],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            logging.info("Error occurred while building the custom data dataframe")
            raise CustomException(e, sys)
