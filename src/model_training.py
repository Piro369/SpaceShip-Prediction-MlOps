import os
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold,cross_validate
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
import joblib
import mlflow


def load_dataset(path):
    X_train = pd.read_parquet(os.path.join(path,'X_train_transformed.parquet'))
    X_test = pd.read_parquet(os.path.join(path,'X_test_transformed.parquet'))
    y_train = pd.read_parquet(os.path.join(path,'y_train.parquet')).squeeze()
    y_test = pd.read_parquet(os.path.join(path,'y_test.parquet')).squeeze()

    print('Dataset Loaded')

    return X_train,X_test,y_train,y_test


def Objective_Function(trial,X_train,y_train):
    with mlflow.start_run(nested=True,run_name=f"Trial_{trial.number}"):

        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'num_leaves': trial.suggest_int('num_leaves', 10, 50), 
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 5),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.8),
            'random_state': 42,
            'verbose': -1
        }

        if param['num_leaves'] > (2 ** param['max_depth']):
            param['num_leaves'] = 2 ** param['max_depth']

        model = LGBMClassifier(**param)
        
        cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            scoring='accuracy',
            cv=cv_strategy,
            return_train_score=True,
            n_jobs=-1
        )

        train_score_mean = cv_results['train_score'].mean()
        val_score_mean = cv_results['test_score'].mean()

        trial.set_user_attr('train_score_mean', train_score_mean)
        trial.set_user_attr('train_score_std', cv_results['train_score'].std())
        trial.set_user_attr('test_score_std', cv_results['test_score'].std())
        trial.set_user_attr('overfitting_gap', train_score_mean - val_score_mean)

        mlflow.log_params(param)
        mlflow.log_metric('val_accuracy',val_score_mean)
        mlflow.log_metric("train_accuracy", train_score_mean)
        mlflow.log_metric('test_score_std',cv_results['test_score'].std())

        
        return val_score_mean



def train_best_model(X_train,y_train,n=30):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial:Objective_Function(trial,X_train,y_train),n_trials=n)

    best_params = study.best_params
    model = LGBMClassifier(**best_params,random_state=42,n_jobs=-1,verbose=-1)
    model.fit(X_train,y_train)
    print('Model Trained')

    return model

def evaluate_model(model,X_test,y_test):
    y_pred = model.predict(X_test)
    print('Accuracy Score',accuracy_score(y_test,y_pred))

def save_model(model,path):
    os.makedirs(path,exist_ok=True)

    joblib.dump(model,os.path.join(path,'best_model.joblib'))
    print('Model Saved Succesfully')


if __name__ == '__main__':
    input_path = 'data/interim/'
    output_path = 'models/'

    X_train,X_test,y_train,y_test = load_dataset(input_path)

    model = train_best_model(X_train,y_train,100)

    evaluate_model(model,X_test,y_test)

    save_model(model,output_path)