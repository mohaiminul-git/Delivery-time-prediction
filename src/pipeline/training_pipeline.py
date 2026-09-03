from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging


class TrainingPipeline:
    def run(self):
        logging.info("Training pipeline started")

        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        model_trainer = ModelTrainer()
        best_model_name, best_model_score = model_trainer.initiate_model_training(train_arr, test_arr)

        logging.info("Training pipeline completed")
        return best_model_name, best_model_score
