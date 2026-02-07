"""
Preprocessing Pipeline for Telco Customer Churn Prediction.

This script handles:
1. Data Loading and Cleaning.
2. Feature Engineering (Custom Transformation).
3. Data Splitting (Stratified).
4. Preprocessing (Scaling, Encoding).
5. Imbalance Handling (SMOTE).
6. Exporting processed datasets for model training.
"""

import pandas as pd
import numpy as np
import os

os.environ['LOKY_MAX_CPU_COUNT'] = '4' 

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE

# --- CONFIGURATION ---
DATA_PATH = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
TARGET_COL = 'Churn'
RANDOM_STATE = 42
TEST_SIZE = 0.2

class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Feature Engineering Transformer.
    Adds interaction terms and domain-specific flags known to impact churn.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Applies feature engineering transformations.
        
        New Features:
        - Tenure_x_MonthlyCharges: Interaction term.
        - TenureGroup: Binned tenure.
        - TotalServicesEngaged: Count of add-on services.
        - Is_ElectronicCheck: High-risk payment flag.
        - Is_FiberOptic: High-risk internet flag.
        - Is_MonthToMonth: High-risk contract flag.
        """
        X = X.copy()
        
        # 1. Tenure Interaction
        if 'tenure' in X.columns and 'MonthlyCharges' in X.columns:
            X['Tenure_x_MonthlyCharges'] = X['tenure'] * X['MonthlyCharges']
            
        # 2. Tenure Groups (Binning)
        if 'tenure' in X.columns:
            X['TenureGroup'] = pd.cut(X['tenure'], bins=[0, 12, 24, 48, 60, np.inf], labels=[1, 2, 3, 4, 5])
            X['TenureGroup'] = X['TenureGroup'].astype(float).fillna(1)

        # 3. Service Usage Count
        services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
        available_services = [col for col in services if col in X.columns]
        if available_services:
            X['TotalServicesEngaged'] = (X[available_services] == 'Yes').sum(axis=1)
        
        # 4. High Risk Flags
        if 'PaymentMethod' in X.columns:
            X['Is_ElectronicCheck'] = (X['PaymentMethod'] == 'Electronic Check').astype(int)
            automatic_list = ['Bank transfer (automatic)', 'Credit card (automatic)']
            X['Is_AutomaticPayment'] = X['PaymentMethod'].isin(automatic_list).astype(int)

        if 'InternetService' in X.columns:
            X['Is_FiberOptic'] = (X['InternetService'] == 'Fiber optic').astype(int)

        if 'Contract' in X.columns:
            X['Is_MonthToMonth'] = (X['Contract'] == 'Month-to-month').astype(int)
            
        return X

def load_and_clean_data(filepath):
    """Loads raw data and performs initial cleaning (type conversion, log transform)."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    
    # Handle TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Log Transform TotalCharges (skewed)
    df['TotalCharges'] = np.log1p(df['TotalCharges'])
    
    # Encode Target
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    return df

def get_preprocessing_pipeline(numeric_features, categorical_features):
    """Builds a Scikit-Learn ColumnTransformer for scaling and encoding."""
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first')) 
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor

def main():
    print("\n=== Preprocessing Pipeline Start ===\n")
    
    # 1. Load
    df = load_and_clean_data(DATA_PATH)
    
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    # 2. Split
    print("Splitting data (80/20 train/test split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")
    
    # 3. Feature Engineering
    print("Applying Feature Engineering...")
    fe = TelcoFeatureEngineer()
    X_train_eng = fe.fit_transform(X_train)
    X_test_eng = fe.transform(X_test)
    
    # Drop TotalCharges (redundant due to interaction term + potential collinearity)
    for data in [X_train_eng, X_test_eng]:
        if 'TotalCharges' in data.columns:
            data.drop(columns=['TotalCharges'], inplace=True)
    
    # Define Columns
    numeric_cols = [
        'tenure', 'MonthlyCharges', 'TotalServicesEngaged', 
        'Tenure_x_MonthlyCharges', 'Is_MonthToMonth', 
        'Is_AutomaticPayment', 'Is_FiberOptic', 'Is_ElectronicCheck'
    ]
    numeric_cols = [c for c in numeric_cols if c in X_train_eng.columns]
    categorical_cols = [c for c in X_train_eng.columns if c not in numeric_cols]
    
    # 4. Transform
    print("Scaling and Encoding...")
    preprocessor = get_preprocessing_pipeline(numeric_cols, categorical_cols)
    
    X_train_processed = preprocessor.fit_transform(X_train_eng)
    X_test_processed = preprocessor.transform(X_test_eng)
    
    # Retrieve feature names
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        # Fallback for older sklearn versions
        feature_names = [f"feat_{i}" for i in range(X_train_processed.shape[1])]

    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    
    # 5. Handle Imbalance (Train only)
    print(f"Original Churn Rate: {y_train.mean():.2%}")
    print("Applying SMOTE to training data...")
    
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_df, y_train)
    
    print(f"Resampled Churn Rate: {y_train_resampled.mean():.2%}")
    
    # 6. Save
    train_final = pd.concat([X_train_resampled, y_train_resampled.reset_index(drop=True)], axis=1)
    test_final = pd.concat([X_test_df, y_test.reset_index(drop=True)], axis=1)
    
    train_final.to_csv('train_processed.csv', index=False)
    test_final.to_csv('test_processed.csv', index=False)
    
    print("\nPipeline Complete.")
    print("Outputs: 'train_processed.csv', 'test_processed.csv'")

if __name__ == "__main__":
    main()
