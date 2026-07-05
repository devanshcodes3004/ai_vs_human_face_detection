from src.exception import CustomException
from src.logger import logger

import os
import sys


class DataValidation:

    def __init__(self):

        print("Data Validation process started.")
        logger.info("Data Validation process started.")

        self.dataset_dir = "artifacts/data_ingestion/data/human vs ai"

        self.ai_dir = os.path.join(
            self.dataset_dir,
            "Human Faces Dataset",
            "AI-Generated Images"
        )
        
        self.human_dir = os.path.join(
            self.dataset_dir,
            "Human Faces Dataset",
            "Real Images"
        )

        self.allowed_extensions = (".jpg", ".jpeg", ".png")

    def count_images(self, folder_path):

        print(f"Counting images in folder: {folder_path}")
        logger.info(f"Counting images in folder: {folder_path}")

        return len(
            [
                file
                for file in os.listdir(folder_path)
                if file.lower().endswith(self.allowed_extensions)
            ]
        )

    def validate_dataset(self):

        try:

            logger.info("Starting Dataset Validation...")

            if not os.path.exists(self.dataset_dir):
                raise FileNotFoundError(
                    f"Dataset directory not found: {self.dataset_dir}"
                )

            logger.info("Dataset directory found.")

            if not os.path.exists(self.ai_dir):
                raise FileNotFoundError(
                    f"AI Images folder not found: {self.ai_dir}"
                )

            logger.info("AI Images folder found.")

            if not os.path.exists(self.human_dir):
                raise FileNotFoundError(
                    f"Human Images folder not found: {self.human_dir}"
                )

            logger.info("Human Images folder found.")

            ai_count = self.count_images(self.ai_dir)
            human_count = self.count_images(self.human_dir)

            logger.info(f"AI Images: {ai_count}")
            logger.info(f"Human Images: {human_count}")

            if ai_count == 0:
                raise ValueError(
                    "AI Images folder is empty."
                )

            if human_count == 0:
                raise ValueError(
                    "Human Images folder is empty."
                )

            logger.info("Dataset Validation Completed Successfully.")
            print("Dataset Validation Completed Successfully.")

            return True

        except Exception as e:
            raise CustomException(e, sys)