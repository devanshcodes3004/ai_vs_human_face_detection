import os
import sys

import tensorflow as tf

from src.exception import CustomException
from src.logger import logger
from src.models import mobilenet
from src.models.custom_cnn import CustomCNN
from src.models.mobilenet import MobileNetV2Model

from src.components.data_preprocessing import DataPreprocessing


class ModelTrainer1:

    def __init__(self):

        self.epochs = 10

        self.model_path = os.path.join(
            "artifacts",
            "models",
            "mobilenet_v2.keras"
        )

    def compile_model(self, model):

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        logger.info("Model compiled successfully.")

    def save_model(self, model):

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        model.save(self.model_path)

        logger.info(f"Model saved successfully at {self.model_path}")

    def train(self):

        try:

            logger.info("Model Training Started.")
            print("Model Training Started.")

            
            preprocessing = DataPreprocessing()

            train_dataset, validation_dataset = preprocessing.create_datasets()

            logger.info("Datasets loaded successfully.")

        
            mobilenet = MobileNetV2Model()

            model = mobilenet.build_model()

            logger.info("MobileNetV2 model created.")

            
            self.compile_model(model)
            print("Model compiled successfully.")
            print("Starting model training...")
            logger.info("Starting model training...")

        
            history = model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=self.epochs
            )

            print("Model training completed successfully.")

            logger.info("Saving model...")
            self.save_model(model)

            return history

        except Exception as e:
            raise CustomException(e, sys)