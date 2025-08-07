from dataclasses import dataclass
import os,sys
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path= os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting train and test input data')

            x_train, y_train, x_test,y_test= (
                train_array[:,: -1],
                train_array[:,-1],
                test_array[:, :-1],
                test_array[:,-1]
            )

            # A dictionary comprising of list of models
            models = {
            'Logistic Regression' : LogisticRegression(solver='lbfgs',max_iter=1500,random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42,max_depth=4, criterion='gini'),
            'Random Forest': RandomForestClassifier(random_state=42),
            'K Nearest Neighbours': KNeighborsClassifier(n_neighbors=3),
            'Naive Bayes': GaussianNB(),
            'Support Vector Machine': SVC(random_state = 42)
            }

            best_model, model_report= evaluate_models(
                x_train= x_train,
                y_train= y_train,
                x_test= x_test,
                y_test= y_test,
                models= models
                )
            logging.info(f"Best model identified-->{best_model}. Saving to disk")
            
            

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )

            y_pred= best_model.predict(x_test)

            f1_weighted= f1_score(y_test, y_pred, average='weighted')

            return f1_weighted
        
        except Exception as e:
            raise CustomException(e,sys)
 
        
