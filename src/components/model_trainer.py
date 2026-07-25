import os 
import sys 
from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

from src.utils import save_object,evaluate_models
from src.logger import logging
from src.exception import CustomException

@dataclass
class ModelTrainerConfig:
    train_model_file_path=os.path.join('artifact',"model.pkl")

class model_trainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("splitting training and test input data")

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )
            models={
                "random forest":RandomForestRegressor(),
                "decision tree":DecisionTreeRegressor(),
                "gradient boost":GradientBoostingRegressor(),
                "k_N_reg":KNeighborsRegressor(),
                "XGBoost":XGBRegressor(),
                "cat boost reg":CatBoostRegressor(verbose=False),
                "adaboost REG":AdaBoostRegressor(),
            }
            params = {
                "random forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 20]
                },

                "decision tree": {
                    "max_depth": [5, 10, 15]
                },

                "gradient boost": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [100, 200]
                },

                "k_N_reg": {
                    "n_neighbors": [3, 5, 7]
                },

                "XGBoost": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [100, 200]
                },

                "cat boost reg": {
                    "depth": [4, 6, 8],
                    "iterations": [100, 200]
                },

                "adaboost REG": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1]
                }
            }

            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,param=params)

            #to get best scoring model from dict
            best_model_score =max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            if best_model_score <0.6:
                raise CustomException("no best model found",sys)

            logging.info(f"best model found on bith training and testing dataset ")

            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)

            r2sq=r2_score(y_test,predicted)
            return r2sq


        except Exception as e :
            raise  CustomException(e,sys)

