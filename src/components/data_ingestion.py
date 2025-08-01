import os, sys
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path : str= os.path.join('artifacts', 'train.csv')
    test_data_path : str= os.path.join('artifacts', 'test.csv')
    raw_data_path : str= os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self. ingestion_config= DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion component')
        try:
            df= pd.read_csv(r'notebooks\data\QualityPrediction.csv')
            logging.info('Read data as a dataframe')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info('Train and test data split initiated')
            train_set, test_set= train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index= False, header= True)
            test_set.to_csv(self.ingestion_config.test_data_path, index= False, header= True)

            logging.info('Data ingestion completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e,sys)
if __name__== '__main__':
    obj= DataIngestion()
    train_data, test_data= obj.initiate_data_ingestion()


