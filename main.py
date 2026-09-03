from src.logger import logging
from src.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        best_model_name, best_model_score = pipeline.run()
        print(f"Training complete. Best model: {best_model_name} (R2 score: {best_model_score:.4f})")
    except Exception as e:
        logging.error(f"Training pipeline failed: {e}")
        raise
