import os
import sys

from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


#Run the DataIngestion
if __name__ == '__main__':
    logging.info("Ingestion starts")
    obj = DataIngestion()
    train_data_path,test_data_path = obj.intiate_data_ingestion()
    logging.info("Ingestion completed.............")
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initate_data_transformation(train_data_path,test_data_path)
    logging.info("Model_traininng started")
    model_trainer = ModelTrainer()

    model_trainer.initaite_model_training(train_arr,test_arr)
