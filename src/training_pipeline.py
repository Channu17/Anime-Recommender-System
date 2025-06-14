from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTraining
from config.path_config import *
from utils.common_functions import read_yaml_file


if __name__ == "__main__":
    
    preprocessor = DataPreprocessor(ANIME_LIST_CSV, PROCESSED_DIR)
    preprocessor.run()
    
    model_trainer = ModelTraining(data_path=PROCESSED_DIR)
    model_trainer.train_model()