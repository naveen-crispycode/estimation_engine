import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor,MultiOutputClassifier
from sklearn.metrics import mean_squared_error, hamming_loss





from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ClassificationModelTrainerConfig:
    Classification_model_file_path = os.path.join("artifacts","classification_model.pkl")
class ClassificationModeltrainer:
    def __init__(self):
        self.classification_model_trainer_config = ClassificationModelTrainerConfig()

    def initiate_classification_model_training(self,classification_model_train,classification_model_test):
        try:
            logging.info("splitting the dataset for classification")
            X_train,y_train,X_test,y_test = (
                classification_model_train[:,:373],
                classification_model_train[:,373:],
                classification_model_test[:,:373],
                classification_model_test[:,373:]
            )
            classfication_model = RandomForestClassifier()
            Multi_classifier = MultiOutputClassifier(classfication_model)
            Multi_classifier.fit(X_train,y_train)
            logging.info("Created a Classification model with multi Output Multi output classifier")
            save_object(
                file_path=self.classification_model_trainer_config.Classification_model_file_path,
                obj=Multi_classifier
            )
            classification_prediction = Multi_classifier.predict(X_test)
            Classification_output = pd.DataFrame(classification_prediction)
            #Classification_output.to_csv("/Users/naveen/Desktop/Projects/Space_Matrix/Estimation_Engine/artifacts/classification_prediction.csv")
            classification_score = hamming_loss(classification_prediction,y_test)
            return classification_score
            
            
        except Exception as e:
            raise CustomException(e,sys)




@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Split the training set and test set")
            X_train,y_train,X_test,y_test = (

                train_array[:,:373],
                train_array[:,373:],
                test_array[:,:373],
                test_array[:,373:]
            )
            models =RandomForestRegressor()
            multioutput_model = MultiOutputRegressor(models)
            multioutput_model.fit(X_train, y_train)
            logging.info("Created a model with multi Output regression")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=multioutput_model
            )
            predicted=multioutput_model.predict(X_test)
            predicted_df = pd.DataFrame(predicted)
            predicted_df.to_csv("/Users/naveen/Desktop/Projects/Space_Matrix/Estimation_Engine/artifacts/predicted.csv")
            y_test_df = pd.DataFrame(y_test)
            y_test_df.to_csv("/Users/naveen/Desktop/Projects/Space_Matrix/Estimation_Engine/artifacts/test.csv")
            score = mean_squared_error(y_test, predicted)
            logging.info(f"Random Forest Multi-output Model MSE: {score}")
            return score
            

            
        except Exception as e:
            raise CustomException(e,sys)