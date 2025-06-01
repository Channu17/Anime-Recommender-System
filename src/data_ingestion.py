from src.logger import get_logger
from src.custom_exception import CustomExecption
from config.path_config import *
import os
from google.cloud import storage
import pandas as pd
from utils.common_functions import read_yaml_file


logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config['data_ingestion']
        self.buket_name = self.config['bucket_name']
        self.file_names = self.config['bucket_file_name']
        
        os.makedirs(RAW_DIR, exist_ok = True)
        logger.info(f'Created directory {RAW_DIR} for raw data')
        
    def download_data(self):
        try:
           client = storage.Client()
           bucket = client.bucket(self.buket_name)
           for file_name in self.file_names:
                if file_name == "animelist.csv":
                    blob = bucket.blob(file_name)
                    blob.download_to_filename(os.path.join(RAW_DIR, file_name))
                    
                    data = pd.read_csv(os.path.join(RAW_DIR, file_name), nrows = 5000000)
                    data.to_csv(os.path.join(RAW_DIR, file_name))
                    
                    logger.info(f"downloaded the file {file_name} from bucket {self.buket_name} to {RAW_DIR} which has large data")
                    
                else:
                    blob = bucket.blob(file_name)
                    blob.download_to_filename(os.path.join(RAW_DIR, file_name))
                    
                    logger.info(f"downloaded the file {file_name} from bucket {self.buket_name} to {RAW_DIR}")
        except Exception as e:
            logger.error(f"Error in downloading data from bucket {self.buket_name}: {e}")
            raise CustomExecption("failed to download the data", e)
        
    def run(self):
        try:
            logger.info("Starting data ingestion process")
            self.download_data()
            logger.info("Data ingestion process completed successfully")
        except Exception as e:
            logger.error(f"Error in data ingestion process: {e}")
            raise CustomExecption("Data ingestion failed", e)

if __name__ == "__main__":
     data_ingestion = DataIngestion(read_yaml_file(CONFIG_YAML))
     data_ingestion.run()
     