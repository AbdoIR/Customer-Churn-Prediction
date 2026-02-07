"""
Model Training and Optimization Pipeline.

This script:
1. Loads processed data (train/test).
2. Defines hyperparameter spaces for multiple models.
3. Optimizes models using RandomizedSearchCV (optimizing for ROC-AUC).
4. Evaluates the best model on the test set.
5. Saves the best performing model to disk.

Models Evaluated:
- Logistic Regression
- Random Forest
- XGBoost
- Gradient Boosting
"""

import pandas as pd
import numpy as np
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4' 

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

# --- CONFIGURATION ---
TRAIN_PATH = 'train_processed.csv'
TEST_PATH = 'test_processed.csv'
TARGET_COL = 'Churn'
RANDOM_STATE = 42
MODELS_DIR = 'models'
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pkl')

def load_data():
    """Loads processed training and testing datasets."""
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        raise FileNotFoundError("Processed data files not found. Run preprocessing_pipeline.py first.")
    
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    
    return X_train, y_train, X_test, y_test

def get_models_and_params():
    """
    Defines the models and their hyperparameter distributions for RandomizedSearch.
    Includes class_weight='balanced' to handle class imbalance.
    """
    models = {
        "Logistic Regression": (
            LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced'),
            {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['lbfgs', 'liblinear']
            }
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
            {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        ),
        "XGBoost": (
            XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss'),
            {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.6, 0.8, 1.0],
                'scale_pos_weight': [1, 3, 5, 7] # Critical for imbalance
            }
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.8, 1.0]
            }
        )
    }
    return models

def train_and_optimize(X_train, y_train, X_test, y_test):
    """
    Runs RandomizedSearchCV for each model, evaluates on test set,
    and returns the best overall model based on ROC-AUC.
    """
    models = get_models_and_params()
    
    results = {}
    best_overall_score = 0
    best_overall_model_name = ""
    best_overall_model = None
    
    print(f"{'Model':<25} | {'ROC-AUC':<10} | {'Accuracy':<10} | {'Status'}")
    print("-" * 65)
    
    for name, (model, params) in models.items():
        # Hyperparameter Tuning
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=20, 
            scoring='roc_auc', # Optimizing for Discrimination not Accuracy
            cv=3,
            verbose=0,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        # Fit
        search.fit(X_train, y_train)
        
        # Best model from search
        best_estimator = search.best_estimator_
        
        # Evaluate on Test Set
        y_pred = best_estimator.predict(X_test)
        
        # Get probabilities for ROC-AUC
        if hasattr(best_estimator, "predict_proba"):
            y_prob = best_estimator.predict_proba(X_test)[:, 1]
        else:
            y_prob = best_estimator.decision_function(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        
        results[name] = {
            "accuracy": acc, 
            "roc_auc": roc, 
            "model": best_estimator,
            "best_params": search.best_params_
        }
        
        print(f"{name:<25} | {roc:.4f}     | {acc:.4f}     | Optimized")
        
        # Update best overall based on ROC-AUC
        if roc > best_overall_score:
            best_overall_score = roc
            best_overall_model_name = name
            best_overall_model = best_estimator
            
    print("-" * 65)
    print(f"Best Model: {best_overall_model_name} with ROC-AUC: {best_overall_score:.4f}")
    
    # Detailed report for best model
    print(f"\n--- Detailed Report for {best_overall_model_name} ---")
    print(f"Best Params: {results[best_overall_model_name]['best_params']}")
    
    y_pred_best = best_overall_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_best))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_best))
    
    return best_overall_model

def save_model(model):
    """Saves the trained model to disk."""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    joblib.dump(model, BEST_MODEL_PATH)
    print(f"\nBest model saved to {BEST_MODEL_PATH}")

def main():
    print("\n=== Model Training Pipeline Start ===\n")
    print("--- 1. Loading Data ---")
    try:
        X_train, y_train, X_test, y_test = load_data()
        print(f"Train features: {X_train.shape}, Test features: {X_test.shape}")
        
        print("\n--- 2. Training & Hyperparameter Optimization ---")
        print("Running RandomizedSearchCV (n_iter=20) on 4 models...")
        print("Metric: ROC-AUC (prioritizing discrimination over accuracy)")
        
        best_model = train_and_optimize(X_train, y_train, X_test, y_test)
        
        print("\n--- 3. Saving Best Model ---")
        save_model(best_model)
        
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
