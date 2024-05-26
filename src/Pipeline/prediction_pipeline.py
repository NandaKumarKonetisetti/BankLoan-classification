import os
import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            logging.info(f"features are :{features}")
            data_scaled = preprocessor.transform(features)
            predict = model.predict(data_scaled)
            logging.info(f"Predicted data is {predict}")
            return predict

        except Exception as e:
            logging.error("Exception occured in prediction",e,sys)
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(
            self,
            age:int,
            experience:int,
            income:int,
            family:int,
            ccavg:float,
            education:int,
            mortgage:float,
            securitiesAcc:int,
            CdAcc:int,
            Online:int,
            Creditcard):
        self.age = age
        self.experience = experience
        self.income = income
        self.family = family
        self.ccavg = ccavg
        self.education = education
        self.mortgage = mortgage
        self.securitiesAcc = securitiesAcc
        self.CdAcc = CdAcc
        self.Online = Online
        self.Creditcard = Creditcard

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Age':[self.age],
                'Experience':[self.experience],
                'Income':[self.income],
                'Family':[self.family],
                'CCAvg':[self.ccavg],
                'Education':[self.education],
                'Mortgage':[self.mortgage],
                'Securities Account':[self.securitiesAcc],
                'CD Account':[self.CdAcc],
                'Online':[self.Online],
                'CreditCard':[self.Creditcard]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Data Gathered for prediction")
            return df

        except Exception as e:
            logging.error(f"error while getting data {e,sys}")
            raise CustomException(e,sys)