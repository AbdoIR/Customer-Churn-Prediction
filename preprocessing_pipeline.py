import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE

# --- CONFIGURATION ---
DATA_PATH = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
TARGET_COL = 'Churn'
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- CUSTOM TRANSFORMER FOR FEATURE ENGINEERING ---
class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        available_services = [col for col in services if col in X.columns]
        if available_services:
            X['TotalServicesEngaged'] = (X[available_services] == 'Yes').sum(axis=1)
        
        # Add epsilon to avoid div by zero
        if 'MonthlyCharges' in X.columns and 'TotalCharges' in X.columns:
            X['Monthly_to_Total_Ratio'] = X['MonthlyCharges'] / (X['TotalCharges'] + 0.01)

        if 'Contract' in X.columns:
            X['Is_MonthToMonth'] = (X['Contract'] == 'Month-to-month').astype(int)

        if 'PaymentMethod' in X.columns:
            automatic_list = ['Bank transfer (automatic)', 'Credit card (automatic)']
            X['Is_AutomaticPayment'] = X['PaymentMethod'].isin(automatic_list).astype(int)
            
        return X

# --- HELPER FUNCTIONS ---

def load_and_clean_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    return df

def get_preprocessing_pipeline(numeric_features, categorical_features):
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())
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
    print("--- 1. Loading and Cleaning Data ---")
    df = load_and_clean_data(DATA_PATH)
    
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    print("--- 2. Splitting Data (Train/Test) ---")
    # Stratify ensure class distribution matches in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
    
    print("--- 3. Feature Engineering ---")
    fe = TelcoFeatureEngineer()
    X_train_eng = fe.fit_transform(X_train)
    X_test_eng = fe.transform(X_test)
    
    X_train_eng = fe.fit_transform(X_train)
    X_test_eng = fe.transform(X_test)
    
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalServicesEngaged', 'Monthly_to_Total_Ratio', 'Is_MonthToMonth', 'Is_AutomaticPayment']
    numeric_cols = [c for c in numeric_cols if c in X_train_eng.columns]
    
    categorical_cols = [c for c in X_train_eng.columns if c not in numeric_cols]
    
    print(f"Numeric Features: {len(numeric_cols)}")
    print(f"Categorical Features: {len(categorical_cols)}")
    
    print("--- 4. Preprocessing (Scaling + Encoding) ---")
    preprocessor = get_preprocessing_pipeline(numeric_cols, categorical_cols)
    
    # Fit on TRAIN, Transform TRAIN and TEST
    X_train_processed = preprocessor.fit_transform(X_train_eng)
    X_test_processed = preprocessor.transform(X_test_eng)
    
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names = [f"feat_{i}" for i in range(X_train_processed.shape[1])]

    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    
    print("--- 5. Handling Imbalance (SMOTE on Train ONLY) ---")
    print(f"Original Train Churn Rate: {y_train.mean():.2%}")
    
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_df, y_train)
    
    print(f"Resampled Train Churn Rate: {y_train_resampled.mean():.2%}")
    print(f"New Train Shape: {X_train_resampled.shape}")
    
    print(f"New Train Shape: {X_train_resampled.shape}")
    
    # --- 6. Saving ---
    train_final = pd.concat([X_train_resampled, y_train_resampled.reset_index(drop=True)], axis=1)
    test_final = pd.concat([X_test_df, y_test.reset_index(drop=True)], axis=1)
    
    train_final.to_csv('train_processed.csv', index=False)
    test_final.to_csv('test_processed.csv', index=False)
    
    print("\n--- Success! ---")
    print("Files saved: 'train_processed.csv', 'test_processed.csv'")

if __name__ == "__main__":
    main()
