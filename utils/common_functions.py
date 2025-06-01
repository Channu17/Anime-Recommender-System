import os
from src.custom_exception import CustomExecption
from src.logger import get_logger
import yaml

logger = get_logger(__name__)


def read_yaml_file(file_path):
    try:
        if not os.path.exists(file_path):
            logger.error(f"YAML file not found at {file_path}")
        with open(file_path, 'r') as file:
            config  = yaml.safe_load(file)
        
        logger.info(F"YAML file read successfully")
        return config
    except Exception as e:
        logger.error(f"Error reading YAML file")
        raise CustomExecption("Failed to read YAML file", e)
