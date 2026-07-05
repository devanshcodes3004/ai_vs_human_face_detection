import os
import sys
import numpy as np
import tensorflow as tf

from src.exception import CustomException
from src.logger import logger


class PredictionPipeline:

    def __init__(self, model_path):

        self.model_path = model_path
        self.image_size = (224, 224)

        logger.info("Loading trained model...")
        self.model = tf.keras.models.load_model(self.model_path)
        logger.info("Model loaded successfully.")

    def preprocess_image(self, image_path):

        try:

            logger.info(f"Loading image: {image_path}")

            image = tf.keras.utils.load_img(
                image_path,
                target_size=self.image_size
            )

            image = tf.keras.utils.img_to_array(image)

            image = image / 255.0

            image = np.expand_dims(image, axis=0)

            logger.info("Image preprocessing completed.")

            return image

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, image_path):

        try:

            image = self.preprocess_image(image_path)

            prediction = self.model.predict(image, verbose=0)

            confidence = float(prediction[0][0])

            if confidence <= 0.5:
                label = "AI Generated"
                display_confidence = 1-confidence
            else:
                label = "Real"
                display_confidence = confidence

            logger.info(
                f"Prediction: {label} | Confidence: {display_confidence:.4f}"
            )

            return {
                "prediction": label,
                "confidence": display_confidence
            }

        except Exception as e:
            raise CustomException(e, sys)