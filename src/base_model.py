from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding,Dot, Flatten, Activation, BatchNormalization, Dropout  
from utils.common_functions import read_yaml_file
from src.custom_exception import CustomExecption
from src.logger import get_logger

logger =  get_logger(__name__)


class BaseModel:
    def __init__(self, config_path: str):
        try:
            self.config = read_yaml_file(config_path)
            logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            raise CustomExecption("Error loading configuration file", e)
        
    def RecommenderNet(self, n_users: int, n_anime: int):
        try:
            embedding_size = self.config['model']['embedding_size']

            user = Input(name="user",shape=[1])

            user_embedding = Embedding(name="user_embedding",input_dim=n_users,output_dim=embedding_size)(user)

            anime = Input(name="anime",shape=[1])

            anime_embedding = Embedding(name="anime_embedding",input_dim=n_anime,output_dim=embedding_size)(anime)

            x = Dot(name="dot_product" , normalize=True , axes=2)([user_embedding,anime_embedding])

            x = Flatten()(x)

            x = Dense(1,kernel_initializer='he_normal')(x)
            x = BatchNormalization()(x)
            x = Activation("sigmoid")(x)

            model = Model(inputs=[user,anime], outputs=x)
            model.compile(loss=self.config['model']['loss'],
                          metrics=self.config['model']['metrics'],
                          optimizer=self.config['model']['optimizer'])
            logger.info("RecommenderNet model built successfully")
            return model
        except Exception as e:
            logger.error(f"Error in building the RecommenderNet model: {e}")
            raise CustomExecption("Error in building the RecommenderNet model", e)
        
    