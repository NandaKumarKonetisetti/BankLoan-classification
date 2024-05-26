import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

import os
import sys 
from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_models,save_object


from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_path :str = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initaite_model_training(self,train_arr,test_arr):
        try:
            logging.info("Model Training has been intialized")

            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models = {
                "logisticreg":LogisticRegression(),
                "kneighbhourclassifier":KNeighborsClassifier(),
                # "multinomialNB":MultinomialNB(),
                # "complementNB":ComplementNB(),
                "bernouali" : BernoulliNB(),
                "Gaussian" : GaussianNB()
                 
            }
            
            model_report , matrix = evaluate_models(X_train,y_train,X_test,y_test,models)
            logging.info("*"*90)
            logging.info(f"Model Report : {model_report}")
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            logging.info(f"best model name is {best_model} and its accuracy {best_model_score}")
            
            save_object(file_path=self.model_trainer_config.trained_model_path,obj=best_model)
            

        except Exception as e:
            logging.error(f"Exception occurred while training the mode {e,sys}")
            raise CustomException(e,sys)
