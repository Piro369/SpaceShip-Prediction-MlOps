from pathlib import Path
import sys
import joblib
from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel
import traceback
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

def safe_log1p(X):
    X = np.asarray(X, dtype=np.float64)
    return np.log1p(np.nan_to_num(X))

class StringExtractionTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            cols = getattr(self, "feature_names_in_", None)
            if cols is not None:
                X = pd.DataFrame(X, columns=cols)
            else:
                # Fallback if no names stored: create string headers
                X = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
        else:
            X = X.copy()
        
        # 1. Clean ALL object/string columns
        for col in X.select_dtypes(include=['object', 'string']).columns:
            X[col] = X[col].replace({'nan': np.nan, 'None': np.nan, '': np.nan})
            X[col] = X[col].mask(X[col].isna(), np.nan)

        # 2. String extractions
        if 'PassengerId' in X.columns:
            X['Passengerno'] = X['PassengerId'].astype(str).str.split('_').str.get(1)
            X['Passengerno'] = X['Passengerno'].replace({'nan': np.nan, 'None': np.nan})
            
        if 'Cabin' in X.columns:
            cabin_split = X['Cabin'].astype(str).str.split('/', expand=True).reindex(columns=[0, 1, 2])
            X[['Deck', 'Num', 'Side']] = cabin_split.values
            X['Num'] = pd.to_numeric(X['Num'], errors='coerce')
            X[['Deck', 'Side']] = X[['Deck', 'Side']].replace({'nan': np.nan, 'None': np.nan})
            
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", [])
        cols = list(input_features)
        if 'PassengerId' in cols and 'Passengerno' not in cols:
            cols.append('Passengerno')
        if 'Cabin' in cols:
            for new_col in ['Deck', 'Num', 'Side']:
                if new_col not in cols:
                    cols.append(new_col)
        return np.array(cols, dtype=object)


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            cols = getattr(self, "feature_names_in_", None)
            if cols is not None:
                X = pd.DataFrame(X, columns=cols)
            else:
                # Fallback if no names stored: create string headers
                X = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
        else:
            X = X.copy()
        
        # TotalBill
        bill_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        existing_bill_cols = [c for c in bill_cols if c in X.columns]
        
        if existing_bill_cols:
            X[existing_bill_cols] = X[existing_bill_cols].apply(pd.to_numeric, errors='coerce')
            X['TotalBill'] = X[existing_bill_cols].sum(axis=1)
        
        # AgeGroup
        bins = [0, 12, 19, 35, 59, 150]
        labels = ['Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']
        
        if 'Age' in X.columns:
            age_series = pd.to_numeric(X['Age'], errors='coerce')
            X['AgeGroup'] = pd.cut(
                age_series, 
                bins=bins, 
                labels=labels, 
                right=True, 
                include_lowest=True
            )
            X['AgeGroup'] = X['AgeGroup'].astype(object).fillna("Adult").astype(str)
            
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", [])
        cols = list(input_features)
        bill_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        if any(c in cols for c in bill_cols) and 'TotalBill' not in cols:
            cols.append('TotalBill')
        if 'Age' in cols and 'AgeGroup' not in cols:
            cols.append('AgeGroup')
        return np.array(cols, dtype=object)


class PredictionRequest(BaseModel):
    PassengerId: str
    HomePlanet: str
    CryoSleep: str
    Cabin: str
    Destination: str
    Age: float
    VIP: str
    RoomService: float
    FoodCourt: float
    ShoppingMall: float 
    Spa: float
    VRDeck: float
    Name: str

class PredictionResponse(BaseModel):
    prediction: bool
    status: str

app = FastAPI(
    title='SpaceShip Prediction',
    description='API endpoint for serving predictions.',
    version='1.0.0'
)

# Pickle path compatibility redirects
current_module = sys.modules[__name__]
sys.modules["src.preprocessor"] = current_module
sys.modules["__main__"] = current_module

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
PIPELINE_PATH = BASE_DIR / "models" / "preprocessor_pipeline.joblib"

try:
    preprocessor = joblib.load(PIPELINE_PATH)
    model = joblib.load(MODEL_PATH)
    print("SUCCESS: Model and Preprocessor loaded cleanly!")
except Exception as e:
    preprocessor = None
    model = None
    print(f"Failed to load artifacts: {e}")
    traceback.print_exc()

try:
    preprocessor.set_output(transform="pandas")
except Exception:
    pass


@app.get('/')
def health_check():
    if model is None or preprocessor is None:
        return {'status': 'unhealthy', 'model_loaded': model is not None, 'pipeline_loaded': preprocessor is not None}
    return {'status': 'healthy'}


@app.post('/predict', response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model artifact is not loaded.")
    
    try:
        input_data_dict = {
            "PassengerId": [request.PassengerId],
            "HomePlanet": [request.HomePlanet],
            "CryoSleep": [request.CryoSleep],
            "Cabin": [request.Cabin],
            "Destination": [request.Destination],
            "Age": [float(request.Age)],
            "VIP": [request.VIP],
            "RoomService": [float(request.RoomService)],
            "FoodCourt": [float(request.FoodCourt)],
            "ShoppingMall": [float(request.ShoppingMall)],
            "Spa": [float(request.Spa)],
            "VRDeck": [float(request.VRDeck)],
            "Name": [request.Name]
        }

        # Create 1-row DataFrame
        feature_dataset = pd.DataFrame(input_data_dict)

        # 1. Transform raw inputs using preprocessor
        transformed_features = preprocessor.transform(feature_dataset)

        # 2. Predict directly with trained model
        raw_prediction = model.predict(transformed_features)
        final_prediction = bool(raw_prediction[0])

        return PredictionResponse(prediction=final_prediction, status='success')

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=400, detail=f"Prediction Error: {str(e)}"
        )
