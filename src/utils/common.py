from pathlib import Path
import sys
import yaml
from src.logger import logger
from src.exception import CustomException
import os,sys


def read_yaml(file_path: str) -> dict:

    try:
        with open(file_path,"r") as yaml_file:
              logger.info(f"YAML file loaded successfully: {file_path}")
              return yaml.safe_load(yaml_file)
        
    except Exception as e:
        logger.error(f"Failed to read YAML file: {file_path}")
        raise CustomException(e,sys)
    

def create_directories(paths: list):

    try:

        for path in paths:
            os.makedirs(path, exist_ok=True)
            logger.info(f"Directory created successfully: {path}")

    except Exception as e:
        logger.error(f"Failed to create directory: {sys.path}")
        raise CustomException(e, sys)   

