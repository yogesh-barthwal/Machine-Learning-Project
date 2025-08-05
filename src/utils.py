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
    


def evaluate_models(x_train, y_train, x_test, y_test, models)-> pd.DataFrame:

    """
    Fit each model and evaluate on test set. Returns a DataFrame with accuracy and macro/weighted F1.

    """
    try:
        results= []
        for name,model in models.items():
            logging.info(f"Training Model {model}")
            model.fit(x_train,y_train)
            y_pred= model.predict(x_test)
            acc= accuracy_score(y_test,y_pred)
            f1_macro= f1_score(y_test, y_pred,average='macro')
            f1_weighted= f1_score(y_test, y_pred,average='weighted')

            logging.info(f"{name}--> Accuracy: {acc}, F1-macro: {f1_macro}, F1-weighted: {f1_weighted}")

            results.append({
                "Model" : name,
                "Accuracy": round(acc,3),
                "F1-macro": round(f1_macro,3),
                "F1-weighted": round(f1_weighted,3),
            })

            df= pd.DataFrame(results).sort_values(by="Accuracy", ascending= False).reset_index(drop=True)
        return df
    
    except Exception as e:
        logging.error(f"Error in evaluate_models: {e}", exc_info=True)
        raise CustomException(e,sys)