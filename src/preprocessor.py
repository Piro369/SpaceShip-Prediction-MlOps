import os
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

# ADD THIS LINE back to force DataFrames through the pipeline
sklearn.set_config(transform_output="pandas")

def safe_log1p(X):
    X = np.asarray(X, dtype=np.float64)
    return np.log1p(np.nan_to_num(X))



# ---------------------------------------------------------
# 1. Custom Transformer for String Splitting (Passenger & Cabin)
# ---------------------------------------------------------
class StringExtractionTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # 1. Clean ALL object/string columns so missing values are unified to true np.nan
        for col in X.select_dtypes(include=['object', 'string']).columns:
            X[col] = X[col].replace({'nan': np.nan, 'None': np.nan, '': np.nan})
            X[col] = X[col].mask(X[col].isna(), np.nan)

        # 2. String extractions
        if 'PassengerId' in X.columns:
            X['Passengerno'] = X['PassengerId'].astype(str).str.split('_').str.get(1)
            X['Passengerno'] = X['Passengerno'].replace({'nan': np.nan, 'None': np.nan})
            
        if 'Cabin' in X.columns:
            X[['Deck', 'Num', 'Side']] = X['Cabin'].astype(str).str.split('/', expand=True)
            X['Num'] = pd.to_numeric(X['Num'], errors='coerce')
            X[['Deck', 'Side']] = X[['Deck', 'Side']].replace({'nan': np.nan, 'None': np.nan})
            
        return X
        

    def get_feature_names_out(self, input_features=None):
        # Fallback to saved feature names if Scikit-learn doesn't pass them
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", [])
        
        cols = list(input_features)
        
        # Append the new columns we created in transform()
        if 'PassengerId' in cols and 'Passengerno' not in cols:
            cols.append('Passengerno')
            
        if 'Cabin' in cols:
            for new_col in ['Deck', 'Num', 'Side']:
                if new_col not in cols:
                    cols.append(new_col)
                    
        return np.array(cols, dtype=object)


# ---------------------------------------------------------
# 2. Custom Transformer for Feature Engineering (TotalBill & AgeGroup)
# ---------------------------------------------------------
class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # TotalBill
        bill_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        existing_bill_cols = [c for c in bill_cols if c in X.columns]
        
        if existing_bill_cols:
            # Ensure they are numeric after imputation
            X[existing_bill_cols] = X[existing_bill_cols].apply(pd.to_numeric, errors='coerce')
            X['TotalBill'] = X[existing_bill_cols].sum(axis=1)
        
        # AgeGroup
        bins = [0, 12, 19, 35, 59, 150]
        labels = ['Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']
        
        if 'Age' in X.columns:
            # Impute or fill empty bins BEFORE converting to string
            age_series = pd.to_numeric(X['Age'], errors='coerce')
            
            X['AgeGroup'] = pd.cut(
                age_series, 
                bins=bins, 
                labels=labels, 
                right=True, 
                include_lowest=True
            )
            # Use a default category like most frequent ("Adult") or fill properly instead of creating a "None" string
            X['AgeGroup'] = X['AgeGroup'].astype(object).fillna("Adult").astype(str)
            
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", [])
        
        cols = list(input_features)
        bill_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        
        # Append TotalBill if the billing columns existed in the input
        if any(c in cols for c in bill_cols) and 'TotalBill' not in cols:
            cols.append('TotalBill')
            
        # Append AgeGroup if Age existed in the input
        if 'Age' in cols and 'AgeGroup' not in cols:
            cols.append('AgeGroup')
            
        return np.array(cols, dtype=object)


def load_dataset(path):
    X_train = pd.read_parquet(os.path.join(path, 'X_train.parquet'))
    X_test = pd.read_parquet(os.path.join(path, 'X_test.parquet'))
    y_train = pd.read_parquet(os.path.join(path, 'y_train.parquet'))
    y_test = pd.read_parquet(os.path.join(path, 'y_test.parquet'))
    print('Dataset Loaded')
    return X_train, X_test, y_train, y_test


def create_and_save_pipeline(X_train, X_test, path):
    os.makedirs(path, exist_ok=True)

    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Passengerno'] 
    # Add 'Num' back here
    num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num'] 

    # Add 'CryoSleep' back here
    ordinal_cols = ['VIP', 'Side', 'CryoSleep']
    ohe_cols = ['HomePlanet', 'Destination', 'Deck', 'AgeGroup', 'Passengerno']



    imputer_step = ColumnTransformer(
    [
        ('SimpleImputer', SimpleImputer(strategy='most_frequent',missing_values=np.nan), cat_cols),
        ('IterativeImputer', IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=50,max_depth=5),
            min_value=0),num_cols)
    ],
    remainder='passthrough', 
    verbose_feature_names_out=False
    )

    num_pipeline = Pipeline([
        ('log', FunctionTransformer(
            func=safe_log1p, 
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
            ('OrdinalEncoding', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols),
            ('DropColumns','drop',['Cabin','Name','PassengerId'])
        ],
        remainder='passthrough', 
        verbose_feature_names_out=False
    )

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

    print('Now running Pipeline fit_transform...')
    X_train_processed = master_pipeline.fit_transform(X_train)
    X_test_processed = master_pipeline.transform(X_test)

    joblib.dump(master_pipeline, os.path.join(path, 'preprocessor_pipeline.joblib'), compress=3)
    print('Pipeline Saved Successfully')
    return X_train_processed, X_test_processed


def save_processed_data(X_train, X_test, y_train, y_test, path):
    os.makedirs(path, exist_ok=True)
    pd.DataFrame(X_train).to_parquet(os.path.join(path, 'X_train_transformed.parquet'), index=False)
    pd.DataFrame(X_test).to_parquet(os.path.join(path, 'X_test_transformed.parquet'), index=False)
    y_train.to_parquet(os.path.join(path, 'y_train.parquet'), index=False)
    y_test.to_parquet(os.path.join(path, 'y_test.parquet'), index=False)
    print('Preprocessed Datasets Saved Successfully')


if __name__ == '__main__':
    input_path = 'data/processed/'
    output_path = 'data/interim/'
    pipeline_path = 'models/'

    X_train, X_test, y_train, y_test = load_dataset(input_path)
    X_train_transformed, X_test_transformed = create_and_save_pipeline(X_train, X_test, pipeline_path)
    save_processed_data(X_train_transformed, X_test_transformed, y_train, y_test, output_path)