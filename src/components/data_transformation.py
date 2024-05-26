import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,FunctionTransformer

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,absolute_value

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path : str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def check_negative_values(self, transformed_data, columns):
        if transformed_data.ndim != 2:
            raise ValueError("Transformed data should have 2 dimensions.")

        for i, column in enumerate(columns):
            if i < transformed_data.shape[1]:
                negative_values = transformed_data[:, i][transformed_data[:, i] < 0]
            if len(negative_values) > 0:
                logging.error(f"Negative values found in feature '{column}': {negative_values}")
        else:
            logging.error(f"Column index {i} exceeds the number of columns in the transformed data.")


                
    def get_Data_Transformation_object(self):
        try:
            logging.info("Data Transformation Intiated")

            categorical_col = ['Education','Mortgage','Securities Account','CD Account','Online','CreditCard','Family']
            numerical_col = ['Age','Experience','Income']

            logging.info("Pipeline Initiated")

            #numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('absolute',FunctionTransformer(absolute_value)),
                    ('scaler',StandardScaler())
                ]
            )

            
            #Categorical Pipeline
            cat_pipeline = Pipeline(
                steps =[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('scaler',StandardScaler())
                ]
            )
           
            preprocessor = ColumnTransformer([
                    ('num_pipeline',num_pipeline,numerical_col),
                    ('cat_pipeline',cat_pipeline,categorical_col)
                ])
            return preprocessor

        except Exception as e:
            logging.error("Exception occured while initiating Data transformation object")
            raise CustomException(e,sys)
    
    def initate_data_transformation(self,train_path,test_path):
        try:
            #Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test data completed")
            logging.info(f"Train data {train_df.head().to_string()}")
            logging.info(f"test data {test_df.head().to_string()}")

            preprocessing_obj = self.get_Data_Transformation_object()

            target_column_name = "Personal Loan"
            drop_columns = [target_column_name,'ID','ZIP Code']


            input_feature_train_df  = train_df.drop(drop_columns,axis=1)
            target_feature_train_df = train_df[target_column_name]


            input_feature_test_df = test_df.drop(drop_columns,axis =1 )
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"input feature trained df columns :{input_feature_train_df.columns}")

            #Transfomation using Preprocessor object 

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info("Applying Preprocessing object on training and testing datasets...........")

            # # Check for negative values after transformation
            # self.check_negative_values(input_feature_train_arr, input_feature_train_df.columns)
            # self.check_negative_values(input_feature_test_arr, input_feature_test_df.columns)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error(f"Exception occured in the intiate Data transformation {e,sys}")
            raise CustomException(e,sys)



