import sys, os
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path= os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj,file_obj)

        logging.info(f'Object successfuly saved at {file_path}')

    except Exception as e:
        logging.error(f"Failed to save object at {file_path}: {e}")
        raise CustomException(e,sys)
    

param_grids = {
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['lbfgs'],
        'penalty': ['l2'],
        'max_iter': [1000,1500,2000],
    },

    'Decision Tree': {
        'max_depth': [3, 5, 10, None],
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [2, 5, 10]
    },

    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'bootstrap': [True, False]
    },

    'K Nearest Neighbours': {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },

    'Naive Bayes': {
        # GaussianNB has very few tunable parameters
        'var_smoothing': [1e-09, 1e-08, 1e-07]
    },

    'Support Vector Machine': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
}

    


def evaluate_models(x_train, y_train, x_test, y_test, models)-> pd.DataFrame:

    """
    Fit each model and evaluate on test set. Returns a DataFrame with accuracy and macro/weighted F1.

    """
    results= []
    try:
        trained_models = {}

        for name,model in models.items():

            try:
                logging.info(f"Training Model {model}")

                if name in param_grids:
                    logging.info(f"Hyperparameter tuning for {name}")
                    grid_search= GridSearchCV(model,param_grids[name],cv=3,n_jobs=1,scoring='f1_weighted')
                    grid_search.fit(x_train, y_train)
                    best_model= grid_search.best_estimator_
                    logging.info(f"Best params for{name}:{grid_search.best_params_}")

                else:
                    model.fit(x_train,y_train)
                    best_model= model

                trained_models[name] = best_model

                y_pred= best_model.predict(x_test)

                acc= accuracy_score(y_test,y_pred)
                f1_macro= f1_score(y_test, y_pred,average='macro')
                f1_weighted= f1_score(y_test, y_pred,average='weighted')

                logging.info(f"{name}--> Accuracy: {acc}, F1-macro: {f1_macro}, F1-weighted: {f1_weighted}")

                results.append({
                    "Model" : name,
                    "Accuracy": round(acc,3),
                    "F1-macro": round(f1_macro,3),
                    "F1-weighted": round(f1_weighted,3),
                    "Best_Estimator" :best_model,
                })

            except Exception as model_err:
                logging.error(f"Failed to evaluate model {name}: {model_err}", exc_info=True)



        df= pd.DataFrame(results).sort_values(by="F1-weighted", ascending= False).reset_index(drop=True)
        best_model_object = df.loc[0, "Best_Estimator"]
    

        return best_model_object, df

        
            
            
        
    except Exception as e:
        logging.error(f"Error in evaluate_models: {e}", exc_info=True)
        raise CustomException(e,sys)
        