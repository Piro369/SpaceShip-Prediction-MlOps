import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import FunctionTransformer, RobustScaler, OneHotEncoder, OrdinalEncoder
import sklearn
import joblib

sklearn.set_config(transform_output="pandas")

# ---------------------------------------------------------
# 1. Custom Transformer for String Splitting (Passenger & Cabin)
# ---------------------------------------------------------
class StringExtractionTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if 'PassengerId' in X.columns:
            X['Passengerno'] = X['PassengerId'].str.split('_').str.get(1)
            
        if 'Cabin' in X.columns:
            X[['Deck', 'Num', 'Side']] = X['Cabin'].str.split('/', expand=True)
            
        return X

# ---------------------------------------------------------
# 2. Custom Transformer for Feature Engineering (TotalBill & AgeGroup)
# ---------------------------------------------------------
class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # TotalBill
        bill_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        # Ensure they are numeric after imputation
        X[bill_cols] = X[bill_cols].apply(pd.to_numeric, errors='coerce')
        X['TotalBill'] = X[bill_cols].sum(axis=1)
        
        # AgeGroup
        bins = [0, 12, 19, 35, 59, 150]
        labels = ['Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']
        
        if 'Age' in X.columns:
            X['AgeGroup'] = pd.cut(
                pd.to_numeric(X['Age']), 
                bins=bins, 
                labels=labels, 
                right=True, 
                include_lowest=True
            )
            # Convert categorical back to string for OneHotEncoder compatibility
            X['AgeGroup'] = X['AgeGroup'].astype(str)
            
        return X

def load_dataset(path):
    X_train = pd.read_csv(path+'X_train.csv',index_col=0)
    X_test = pd.read_csv(path+ 'X_test.csv',index_col=0)
    y_train = pd.read_csv(path+'y_train.csv',index_col=0)
    y_test = pd.read_csv(path+'y_test.csv',index_col=0)

    print('Dataset Loaded')

    return X_train,X_test,y_train,y_test

def create_and_save_pipeline(X_train,X_test,path):

    # (Make sure you define these lists based on your raw dataset columns)
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Passengerno'] 
    num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    ohe_cols = ['HomePlanet', 'Destination', 'Deck', 'AgeGroup', 'Passengerno']
    ordinal_cols = ['VIP', 'Side']

    # ---------------------------------------------------------
    # 4. Building the Individual Pipeline Steps
    # ---------------------------------------------------------
    imputer_step = ColumnTransformer(
        [
            ('SimpleImputer', SimpleImputer(strategy='most_frequent'), cat_cols),
            ('IterativeImputer', IterativeImputer(estimator=ExtraTreesRegressor()), num_cols)
        ],
        remainder='passthrough', 
        verbose_feature_names_out=False
    )

    num_pipeline = Pipeline([
        ('log', FunctionTransformer(
            func=np.log1p, 
            inverse_func=np.expm1, 
            validate=False, 
            feature_names_out="one-to-one"
        )),
        ('scaler', RobustScaler())
    ])

    # We dynamically add 'TotalBill' because it was created in the FeatureEngineering step
    scaler_step = ColumnTransformer(
        [
            ('num_preprocess', num_pipeline, num_cols + ['TotalBill'])
        ], 
        remainder='passthrough', 
        verbose_feature_names_out=False
    )

    encoder_step = ColumnTransformer(
        [
            ('OHE', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), ohe_cols),
            ('OrdinalEncoding', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols)
        ],
        remainder='passthrough', 
        verbose_feature_names_out=False
    )
    
    print('Now running Pipeline')
    # ---------------------------------------------------------
    # 5. The Final Master Pipeline
    # ---------------------------------------------------------
    master_pipeline = Pipeline([
        ('string_extractor', StringExtractionTransformer()),
        ('imputer', imputer_step),
        ('feature_engineer', FeatureEngineeringTransformer()),
        ('scaler', scaler_step),
        ('encoder', encoder_step)
    ])

    # Example Usage:
    X_train_processed = master_pipeline.fit_transform(X_train)
    X_test_processed = master_pipeline.transform(X_test)

    joblib.dump(master_pipeline,path+'preprocessor_pipeline.joblib')
    print('Pipeline Saved Succesfully')
    
    return X_train_processed,X_test_processed


def save_processed_data(X_train,X_test,y_train,y_test,path):
    X_train.to_csv(path+'X_train_transformed.csv')
    X_test.to_csv(path+'X_test_transformed.csv')
    y_train.to_csv(path+'y_train.csv')
    y_test.to_csv(path+'y_test.csv')

    print('Preprocessed Datasets Saved Sucessfully')


if __name__ == '__main__':
    input_path = '../data/processed/'
    output_path = '../data/interim/'
    pipeline_path = '../models/'

    X_train,X_test,y_train,y_test = load_dataset(input_path)

    X_train_transformed,X_test_transformed = create_and_save_pipeline(X_train,X_test,pipeline_path)

    save_processed_data(X_train_transformed,X_test_transformed,y_train,y_test,output_path)


