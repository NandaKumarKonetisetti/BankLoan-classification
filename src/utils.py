import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import classification_report,confusion_matrix,precision_score,jaccard_score,recall_score,f1_score,accuracy_score

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        logging.error(f"Got Exception while saving a pickle file {e,sys}")
        raise CustomException(e,sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models):
    try :
        report = {}
        matrix = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            #Train the model
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            
            model_score = accuracy_score(y_test,y_pred)
            conf_matrix = confusion_matrix(y_test,y_pred)
            report[list(models.keys())[i]]=model_score
            matrix[list(models.keys())[i]] =   conf_matrix          
        return report,matrix


    except Exception as e:
        logging.error(f"Error occured while evaluating models {e,sys} ")
        raise CustomException(e,sys)
    
def absolute_value(x):
    return np.abs(x)

def load_object(file_path):
    try:
        with open(file_path,"rb") as obj:
           return pickle.load(obj)
    except Exception as e:
        logging.error(f"Error occured while loading a model object {e,sys}")
        raise CustomException(e,sys)