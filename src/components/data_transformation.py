import sys, os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FunctionTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path= os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig   ()
        
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation

        '''
        try: 
            cols_for_log_trans= [
                                'fixed acidity',
                                'volatile acidity',
                                'citric acid',
                                'residual sugar',
                                'chlorides',
                                'free sulfur dioxide',
                                'total sulfur dioxide',
                                'sulphates',
                                'alcohol'
                                ]
            
            cols_wo_log_trans= ['density','pH']

            # 'fixed acidity', 'free sulfur dioxide' are to be dropped
            
            log_pipeline= Pipeline(                                         
                steps=[
                ('imputer', SimpleImputer(strategy='median')), #The dataset contains numerical features alone
                ('log_transformation', FunctionTransformer(func= np.log1p, feature_names_out='one-to-one')),
                ('scaler', StandardScaler()), 
            ],
            )

            logging.info('Log Transformation performed for skewed features')
            no_log_pipeline= Pipeline(                                         
                steps=[
                ('imputer', SimpleImputer(strategy='median'),), #The dataset contains numerical features alone
                ('scaler', StandardScaler()), 
            ],
            )

            preprocessor= ColumnTransformer(
                transformers= [
                    ('log_transformation',log_pipeline, cols_for_log_trans),
                    ('wo_log_transformation', no_log_pipeline, cols_wo_log_trans)
                ],
                remainder= 'drop'
            )
            logging.info("'fixed acidity', 'free sulfur dioxide' dropped owing to multi collinearity")


            return preprocessor
        except Exception as e:
            raise  CustomException(e,sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df= pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)

            logging.info('Train and Test data read')
            logging.info('Obtaining preprocessing object')

            preprocessing_obj= self.get_data_transformer_object()

            target_column_name= 'quality'

            input_feature_train_df= train_df.drop([target_column_name], axis=1)
            target_feature_train_df= train_df[target_column_name]

            input_feature_test_df= test_df.drop([target_column_name], axis=1)
            target_feature_test_df= test_df[target_column_name]

            logging.info('Applying preprocessing on train and test dataframe')

            input_feature_train_arr= preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr= preprocessing_obj.transform(input_feature_test_df)

            train_arr= np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr= np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info('Saving preprocessing object')

            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocessing_obj
                )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path

            )

        except Exception as e:
            raise CustomException(e,sys)
            



