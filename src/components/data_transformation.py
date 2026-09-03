import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config.configuration import (
    FEATURE_ENGINEERED_TEST_DATA_PATH,
    FEATURE_ENGINEERED_TRAIN_DATA_PATH,
    FEATURE_ENGINEERING_OBJ_PATH,
    PREPROCESSED_TEST_DATA_PATH,
    PREPROCESSED_TRAIN_DATA_PATH,
    PREPROCESSING_OBJ_PATH,
)
from src.constants import TARGET_COLUMN
from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_obj


class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        logging.info("Feature engineering initialized")

    def get_distance(self, df, lat1, lon1, lat2, lon2):
        # Ensure latitudes and longitudes are absolute values
        df[lat1] = df[lat1].abs()
        df[lat2] = df[lat2].abs()
        df.drop(df[(df["Restaurant_latitude"] < 8) & (df["Delivery_location_latitude"] < 8)].index, inplace=True)
        p = np.pi / 180
        a = (
            0.5
            - np.cos((df[lat2] - df[lat1]) * p) / 2
            + np.cos(df[lat1] * p) * np.cos(df[lat2] * p) * (1 - np.cos((df[lon2] - df[lon1]) * p)) / 2
        )
        df["distance"] = 12734 * np.arccos(np.clip(a, -1, 1))
        return df

    def extract_time(self, df):
        variables_list = ["Time_Orderd", "Time_Order_picked"]
        df.dropna(subset=["Time_Orderd"], inplace=True)
        for var in variables_list:
            df[var] = df[var].replace(".", ":")
            df[var] = pd.to_datetime(df[var], format="%H:%M", errors="coerce")
            df[var] = df[var].dt.strftime("%H:%M")

            df[f"{var}_hour"] = pd.to_datetime(df[var], format="%H:%M").dt.hour.astype("Int32")
            df[f"{var}_min"] = pd.to_datetime(df[var], format="%H:%M").dt.minute.astype("Int32")

    def extract_city(self, df):
        var = "Delivery_person_ID"
        df["Delivery_city"] = df[var].str.split("RES", expand=True)[0]

    def drop_variables(self, df):
        try:
            columns_to_drop = [
                "ID",
                "Delivery_person_ID",
                "Restaurant_latitude",
                "Restaurant_longitude",
                "Delivery_location_latitude",
                "Delivery_location_longitude",
                "Order_Date",
                "Time_Orderd",
                "Time_Orderd_min",
                "Time_Order_picked",
                "Time_Order_picked_hour",
                "Time_Order_picked_min",
            ]
            df.drop(columns=columns_to_drop, inplace=True)
        except Exception as e:
            raise CustomException(e, sys)

    def transform_feature(self, df):
        try:
            df = self.get_distance(
                df, "Restaurant_latitude", "Restaurant_longitude", "Delivery_location_latitude",
                "Delivery_location_longitude",
            )
            self.extract_time(df)
            self.extract_city(df)
            self.drop_variables(df)
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def fit(self, X, y=None):
        return self

    def __sklearn_is_fitted__(self) -> bool:
        # Stateless transformer: it recomputes features from each input batch
        # rather than learning parameters from training data, so it's always
        # ready to transform. Without this, sklearn's check_is_fitted() can't
        # tell fitted apart from never-fitted (it looks for a trailing-"_"
        # attribute set by fit()) and warns on every .transform() call.
        return True

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        try:
            return self.transform_feature(X)
        except Exception as e:
            raise CustomException(e, sys)


@dataclass
class DataTransformationConfig:
    preprocessing_obj_path: str = PREPROCESSING_OBJ_PATH
    preprocessed_train_data_path: str = PREPROCESSED_TRAIN_DATA_PATH
    preprocessed_test_data_path: str = PREPROCESSED_TEST_DATA_PATH
    feature_engineering_obj_path: str = FEATURE_ENGINEERING_OBJ_PATH


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            road_traffic_density = ["Low", "Medium", "High", "Jam"]
            weather_conditions = ["Sunny", "Cloudy", "Windy", "Fog", "Sandstorms", "Stormy"]

            categorical_columns = ["Type_of_order", "Type_of_vehicle", "Festival", "City", "Delivery_city"]
            ordinal_columns = ["Road_traffic_density", "Weather_conditions"]
            numerical_columns = [
                "Delivery_person_Age", "Delivery_person_Ratings", "Vehicle_condition",
                "multiple_deliveries", "Time_Orderd_hour", "distance",
            ]

            numerical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer()),
                ("scaler", StandardScaler(with_mean=False)),
            ])

            categorical_pipeline = Pipeline(steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ("scaler", StandardScaler(with_mean=False)),
            ])

            ordinal_pipeline = Pipeline(steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(categories=[road_traffic_density, weather_conditions])),
                ("scaler", StandardScaler(with_mean=False)),
            ])

            preprocessor = ColumnTransformer([
                ("numerical", numerical_pipeline, numerical_columns),
                ("categorical", categorical_pipeline, categorical_columns),
                ("ordinal", ordinal_pipeline, ordinal_columns),
            ])
            logging.info("Data transformation object obtained")
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def get_feature_engineering_obj(self):
        try:
            return Pipeline(steps=[("feature_engineering", FeatureEngineering())])
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            fe_obj = self.get_feature_engineering_obj()
            train_data = fe_obj.fit_transform(train_data)
            test_data = fe_obj.transform(test_data)

            os.makedirs(os.path.dirname(FEATURE_ENGINEERED_TRAIN_DATA_PATH), exist_ok=True)
            train_data.to_csv(FEATURE_ENGINEERED_TRAIN_DATA_PATH, index=False, header=True)
            test_data.to_csv(FEATURE_ENGINEERED_TEST_DATA_PATH, index=False, header=True)

            processing_obj = self.get_data_transformation_obj()

            x_train = train_data.drop(columns=TARGET_COLUMN, axis=1)
            y_train = train_data[TARGET_COLUMN]

            x_test = test_data.drop(columns=TARGET_COLUMN, axis=1)
            y_test = test_data[TARGET_COLUMN]

            x_train = processing_obj.fit_transform(x_train)
            x_test = processing_obj.transform(x_test)

            train_arr = np.c_[x_train, np.array(y_train)]
            test_arr = np.c_[x_test, np.array(y_test)]

            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessed_train_data_path), exist_ok=True)
            pd.DataFrame(train_arr).to_csv(
                self.data_transformation_config.preprocessed_train_data_path, index=False, header=True
            )

            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessed_test_data_path), exist_ok=True)
            pd.DataFrame(test_arr).to_csv(
                self.data_transformation_config.preprocessed_test_data_path, index=False, header=True
            )

            save_obj(self.data_transformation_config.feature_engineering_obj_path, fe_obj)
            save_obj(self.data_transformation_config.preprocessing_obj_path, processing_obj)

            logging.info("Data transformation completed")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessing_obj_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
