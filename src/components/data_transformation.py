import sys 
from dataclasses import dataclass
import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler#label,numerical
import os 
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path=os.path.join('artifact',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_tranformer_object(self):#to create an pickel file
        '''
        this function is responsible for data transformation
        
        '''
        try:

            numerical_columns=["writing score","reading score"]

            categorical_columns=[
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]
            
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),#handling the missing value with median as stratagy
                    ("scaler",StandardScaler())

                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one hot encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]


            )

            logging.info(f"numerical columns {numerical_columns}")

            logging.info(f"categorical columns {categorical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("numerical_pipeline",num_pipeline,numerical_columns),
                    ("categorical_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor 

            

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Readong the test and train data is completed ")

            logging.info("obtaiing the preprocessor object")


            preprocessor_object = self.get_data_tranformer_object()

            target_column_name="math score"
            numerical_column=["writing score","reading score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name])
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name])
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"appplying the processing object on training dataframe and testing data frame "
                )
            input_feature_train_array=preprocessor_object.fit_transform(input_feature_train_df)
            input_feature_test_array=preprocessor_object.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_array,np.array(target_feature_train_df)
            ]

            test_arr=np.c_[
                input_feature_test_array,np.array(target_feature_test_df)
            ]

            logging.info("savong preprocessor object ")

            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessor_object
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
            



