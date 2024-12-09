
import os
import sys
import numpy as np
import pandas as pd
from src.logger import logging
from src.exceptions import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from sklearn.utils import shuffle
from src.components.data_preprocessing import DataTransformation
from src.components.data_preprocessing import DataTransformationConfig
from src.components.models_training import ModelTrainerConfig
from src.components.models_training import ModelTrainer,ClassificationModeltrainer




@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts","train.csv")
    test_data_path: str = os.path.join("artifacts","test.csv")
    raw_data_path : str = os.path.join("artifacts","Imported_data.csv")
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion stage")
        try:
            df = pd.read_csv("data/data.csv",keep_default_na=False, na_values=[])
            logging.info("Read the dataset as a dataframe")
            """Shuffle the dataset"""
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            shuffled_df = shuffle(df,random_state=8).reset_index(drop = True)
            shuffled_df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True,na_rep=None)
            train_set,test_set = train_test_split(shuffled_df,test_size=0.2,random_state=8)
            train_set.to_csv(self.ingestion_config.train_data_path,index = False,header = True)
            test_set.to_csv(self.ingestion_config.test_data_path,index = False,header = True)
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,classification_train_arr,classification_test_arr,_ = data_transformation.initiate_data_transformation(train_data,test_data)
    logging.info("Data is transformed")
    print(train_arr.shape)
    print(test_arr.shape)
    print(classification_train_arr.shape)
    print(classification_test_arr.shape)
    #modeltrainer=ModelTrainer()
    #print(modeltrainer.initiate_model_training(train_arr,test_arr))
    classification_model = ClassificationModeltrainer()
    print("Model Initiated")
    print(classification_model.initiate_classification_model_training(classification_train_arr,classification_test_arr))
    logging.info("Model has been built")


