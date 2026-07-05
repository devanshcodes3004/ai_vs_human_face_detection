from src.logger import logger
from src.exception import CustomException
import os,sys

import tensorflow as tf


class DataPreprocessing:

    import tensorflow as tf

from src.exception import CustomException
from src.logger import logger

import os
import sys


class DataPreprocessing:

    def __init__(self):

        self.dataset_dir = os.path.join(
            "artifacts",
            "data_ingestion",
            "data",
            "human vs ai",
            "Human Faces Dataset"
        )

        self.image_size = (224, 224)
        self.batch_size = 32
        self.validation_split = 0.2
        self.seed = 42

    def create_datasets(self):

        try:

            logger.info("Starting Data Preprocessing...")

            train_dataset = tf.keras.utils.image_dataset_from_directory(
                self.dataset_dir,
                labels="inferred",
                label_mode="binary",
                validation_split=self.validation_split,
                subset="training",
                seed=self.seed,
                image_size=self.image_size,
                batch_size=self.batch_size,
                shuffle=True
            )

            validation_dataset = tf.keras.utils.image_dataset_from_directory(
                self.dataset_dir,
                labels="inferred",
                label_mode="binary",
                validation_split=self.validation_split,
                subset="validation",
                seed=self.seed,
                image_size=self.image_size,
                batch_size=self.batch_size,
                shuffle=True
            )

            logger.info("Dataset loaded successfully.")

            # Normalize pixel values (0-255 → 0-1)
            normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

            train_dataset = train_dataset.map(
                lambda images, labels: (normalization_layer(images), labels)
            )

            validation_dataset = validation_dataset.map(
                lambda images, labels: (normalization_layer(images), labels)
            )

            # Improve performance
            AUTOTUNE = tf.data.AUTOTUNE

            train_dataset = train_dataset.prefetch(AUTOTUNE)
            validation_dataset = validation_dataset.prefetch(AUTOTUNE)

            logger.info("Data preprocessing completed successfully.")

            return train_dataset, validation_dataset

        except Exception as e:
            raise CustomException(e, sys)