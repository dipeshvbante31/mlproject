import os
import sys
import pandas as pd 
import numpy as np
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.exception import CustomException
import sys


def evaluate_models(x_train, y_train, x_test, y_test, models, param):

    try:

        report = {}

        for i in range(len(models)):

            model_name = list(models.keys())[i]
            model = list(models.values())[i]

            para = param[model_name]

            # Hyperparameter tuning
            gs = GridSearchCV(
                estimator=model,
                param_grid=para,
                cv=3,
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )

            gs.fit(x_train, y_train)

            # Best parameters
            model.set_params(**gs.best_params_)

            # Train model with best parameters
            model.fit(x_train, y_train)

            # Predictions
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_score

            print(f"{model_name}")
            print(f"Best Parameters: {gs.best_params_}")
            print("-" * 50)

        return report

    except Exception as e:
        raise CustomException(e, sys)