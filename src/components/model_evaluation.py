import os
import sys
import json

from sklearn import metrics
import numpy as np
import tensorflow as tf
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from src.logger import logger
from src.exception import CustomException
from src.components.data_preprocessing import DataPreprocessing


class ModelEvaluation:

    def __init__(self):

        self.model_path = os.path.join(
            "artifacts",
            "models",
            "mobilenet_v2.keras"
        )

        self.metrics_path = os.path.join(
            "artifacts",
            "evaluation"
        )

    def evaluate(self):

        try:

            logger.info("Starting Model Evaluation...")

            
            os.makedirs(self.metrics_path, exist_ok=True)

        
            preprocessing = DataPreprocessing()

            _, validation_dataset = preprocessing.create_datasets()

            logger.info("Datasets loaded successfully.")

        
            model = tf.keras.models.load_model(self.model_path)

            logger.info("Model Loaded Successfully.")

            
            loss, accuracy = model.evaluate(
                validation_dataset,
                verbose=1
            )

            logger.info(f"Validation Loss : {loss}")
            logger.info(f"Validation Accuracy : {accuracy}")

        
            y_true = []
            y_pred = []

            for images, labels in validation_dataset:

                predictions = model.predict(images, verbose=0)

                predictions = (predictions > 0.5).astype(int)

                y_true.extend(labels.numpy())

                y_pred.extend(predictions.flatten())

            
            accuracy = accuracy_score(y_true, y_pred)

            precision = precision_score(y_true, y_pred)

            recall = recall_score(y_true, y_pred)

            f1 = f1_score(y_true, y_pred)

            report = classification_report(
                y_true,
                y_pred
            )

            logger.info(report)

            metrics = {

                "Accuracy": float(accuracy),
                "Precision": float(precision),
                "Recall": float(recall),
                "F1 Score": float(f1)

            }

            

            joblib.dump(metrics, "metrics.joblib")

            logger.info("Metrics Saved Successfully.")

            print(report)

            return metrics

        except Exception as e:

            raise CustomException(e, sys)