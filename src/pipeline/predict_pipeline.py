import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self)-> None:
        ...


    def predict(self,features):
        try:
            model_path= os.path.join('artifacts','model.pkl')
            preprocessor_path= os.path.join('artifacts','preprocessor.pkl')
            model= load_object(file_path=model_path)
            preprocessor= load_object(file_path=preprocessor_path)
            transformed_data= preprocessor.transform(features)
            preds= model.predict(transformed_data)
            return preds

        except Exception as e:
            raise CustomException(e,sys)    

class CustomData:
    def __init__(self,
                 fixed_acidity: float,
        volatile_acidity: float,
        citric_acid: float,
        residual_sugar: float,
        chlorides: float,
        free_sulfur_dioxide: float,
        total_sulfur_dioxide: float,
        density: float,
        pH: float,
        sulphates: float,
        alcohol: float,
    ):
        self.fixed_acidity = fixed_acidity
        self.volatile_acidity = volatile_acidity
        self.citric_acid = citric_acid
        self.residual_sugar = residual_sugar
        self.chlorides = chlorides
        self.free_sulfur_dioxide = free_sulfur_dioxide
        self.total_sulfur_dioxide = total_sulfur_dioxide
        self.density = density
        self.pH = pH
        self.sulphates = sulphates
        self.alcohol = alcohol

    def get_data_as_dataframe(self) -> pd.DataFrame:
        data_dict = {
            'fixed acidity': [self.fixed_acidity],
            'volatile acidity': [self.volatile_acidity],
            'citric acid': [self.citric_acid],
            'residual sugar': [self.residual_sugar],
            'chlorides': [self.chlorides],
            'free sulfur dioxide': [self.free_sulfur_dioxide],
            'total sulfur dioxide': [self.total_sulfur_dioxide],
            'density': [self.density],
            'pH': [self.pH],
            'sulphates': [self.sulphates],
            'alcohol': [self.alcohol],
        }
        return pd.DataFrame(data_dict)