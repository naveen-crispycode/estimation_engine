import sys
import os
import pandas as pd
from src.exceptions import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor = os.path.join("artifacts","preprocessor.pkl")
            SKU_labels_path = os.path.join("artifacts","SKU_label.csv")
            print("Started_Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor)
            sku_label_list = pd.read_csv(SKU_labels_path)["SKU_Labels"].tolist()
            print("Loaded Model & Preprocessor")
            data_transformed = preprocessor.transform(features)
            prediction = model.predict(data_transformed)
            results = {sku: quantity for sku, quantity in zip(sku_label_list, prediction[0])}
            return results
    
        except Exception as e:
            raise CustomException(e,sys)
        
    def classify(self,features):
        try:
            Classify_model_path = os.path.join("artifacts","classification_model.pkl")
            preprocessor = os.path.join("artifacts","preprocessor.pkl")
            SKU_labels_path = os.path.join("artifacts","SKU_label.csv")
            print("Started_Loading")
            classify_model = load_object(file_path=Classify_model_path)
            preprocessor = load_object(file_path=preprocessor)
            sku_label_list = pd.read_csv(SKU_labels_path)["SKU_Labels"].tolist()
            print("Loaded Model & Preprocessor")
            data_transformed = preprocessor.transform(features)
            SKU_Prediction = classify_model.predict(data_transformed)
            results = {sku: quantity for sku, quantity in zip(sku_label_list, SKU_Prediction[0])}
            non_zero_skus = {sku: outcome for sku, outcome in results.items() if outcome == 1}
            return non_zero_skus
        except Exception as e:
            raise CustomException(e,sys)
    def __call__(self, features, task="predict"):
    
        if task == "predict":
            return self.predict(features)
        elif task == "classify":
            return self.classify(features)
        else:
            raise ValueError(f"Invalid task: {task}. Use 'predict' or 'classify'.")




