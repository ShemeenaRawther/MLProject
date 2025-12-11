import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass #helps to define classes which are mainly used to store values in variables
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str  = os.path.join('artifacts','test.csv')
    raw_data_path: str   = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.injection_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method starts")
        try:
            df= pd.read_csv('src/notebook/data/stud.csv')            
            logging.info("Dataset read as pandas dataframe")

            os.makedirs(os.path.dirname(self.injection_config.train_data_path),exist_ok=True)
            df.to_csv(self.injection_config.raw_data_path,index=False,header=True)
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)    
            train_set.to_csv(self.injection_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.injection_config.test_data_path,index=False,header=True)
            return (self.injection_config.train_data_path,
                    self.injection_config.test_data_path)
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()