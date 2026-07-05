from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.model_trainer import ModelTrainer
from src.components.model_transefer_trainer import ModelTrainer1
from src.components.model_evaluation import ModelEvaluation
from src.pipeline.prediction_pipeline import PredictionPipeline


if __name__ == "__main__":


    # obj = DataIngestion()
    # obj.initiate_data_ingestion()

    # validator = DataValidation()
    # validator.validate_dataset()

    # trainer = ModelTrainer()
    # trainer.train()

    # trainer = ModelTrainer1()
    # trainer.train()

    # evaluator = ModelEvaluation()

    # metrics = evaluator.evaluate()

    predictor = PredictionPipeline(
        model_path="artifacts/models/mobilenet_v2.keras"
    )

    result = predictor.predict(
        "test11.png"
    )

    print(f"Prediction Result: {result} and confidence: {result['confidence']:.2f}")
    print("Pipeline completed successfully.")