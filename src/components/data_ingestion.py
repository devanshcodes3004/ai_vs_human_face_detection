from src.exception import CustomException
from src.logger import logger
import os,sys
from src.utils.common import read_yaml, create_directories
import shutil

class DataIngestion:

    def __init__(self):
        self.source_dir = "data/raw"
        self.destination_dir = "artifacts/data_ingestion/data"


    def initiate_data_ingestion(self):

        logger.info("Data ingestion process started.")

        try:

            if not os.path.exists(self.source_dir):
                logger.error(f"Source directory does not exist: {self.source_dir}")
                raise CustomException(f"Source directory does not exist: {self.source_dir}", sys)
            
            

            if os.path.exists(self.destination_dir):
                logger.info(f"destination directory allready exists:{self.destination_dir}")
                shutil.rmtree(self.destination_dir)
                logger.info(f"destination directory deleted successfully:{self.destination_dir}")
               
            print(f"copying data from {self.source_dir} to {self.destination_dir}")   
            logger.info(f"copying data from {self.source_dir} to {self.destination_dir}")    
            shutil.copytree(self.source_dir,self.destination_dir,dirs_exist_ok=True)

            logger.info(f"data ingestion completed sucessfully from {self.source_dir} to {self.destination_dir}")
            return self.destination_dir

        except Exception as e:
            raise CustomException(e,sys)      



if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()        




