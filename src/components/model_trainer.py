from dataclasses import dataclass
import os,sys
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import accuracy_score, recall_score,confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_score,roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
from sklearn.preprocessing import RobustScaler
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
            'Logistic Regression' : LogisticRegression(),
            'Decision Tree': DecisionTreeClassifier(random_state=42,max_depth=4, criterion='gini'),
            'Random Forest': RandomForestClassifier(random_state=42),
            'K Nearest Neighbours': KNeighborsClassifier(n_neighbors=3),
            'Naive Bayes': GaussianNB(),
            'Support Vector Machine': SVC(random_state = 42)
            }

            model_report: pd.DataFrame= evaluate_models(
                x_train= x_train,
                y_train= y_train,
                x_test= x_test,
                y_test= y_test,
                models= models
                )
            
            
            # print("Model report:")
            # print(model_report)
            # print("Columns:", model_report.columns)
            # print("First row:")
            # print(model_report.iloc[0])


            
            best_idx= model_report['F1-weighted'].idxmax()
            best_row= model_report.loc[best_idx]
            best_model= best_row['Model']
            best_score= best_row['F1-weighted']
            logging.info('Best model Found')

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )

            models[best_model].fit(x_train, y_train)
            predicted= models[best_model].predict(x_test)

            f1_weighted= f1_score(y_test, predicted, average='weighted')

            return f1_weighted
        
        except Exception as e:
            raise CustomException(e,sys)
        
