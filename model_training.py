"""
Model Training and Optimization Pipeline with Visualization.

This script:
1. Loads processed data (train/test).
2. Defines hyperparameter spaces for multiple models.
3. Optimizes models using RandomizedSearchCV (optimizing for ROC-AUC).
4. Evaluates models and selects the best one.
5. Visualizes performance (ROC, PR Curves, Confusion Matrices).
6. Saves the best performing model to disk.

Models Evaluated:
- Logistic Regression
- Random Forest
- XGBoost
- Gradient Boosting
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
os.environ['LOKY_MAX_CPU_COUNT'] = '4' 

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, precision_recall_curve, 
                             average_precision_score, ConfusionMatrixDisplay)
from sklearn.model_selection import RandomizedSearchCV

# --- CONFIGURATION ---
TRAIN_PATH = 'train_processed.csv'
TEST_PATH = 'test_processed.csv'
TARGET_COL = 'Churn'
RANDOM_STATE = 42
MODELS_DIR = 'models'
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pkl')
VISUALIZATIONS_DIR = 'visualisations'

def load_data():
    """Loads processed training and testing datasets."""
    # For demonstration, creating dummy data if files don't exist
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        print("Warning: Data files not found. Generating dummy data for demonstration...")
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        X_train_dummy = pd.DataFrame(X[:800])
        y_train_dummy = pd.Series(y[:800], name=TARGET_COL)
        X_test_dummy = pd.DataFrame(X[800:])
        y_test_dummy = pd.Series(y[800:], name=TARGET_COL)
        return X_train_dummy, y_train_dummy, X_test_dummy, y_test_dummy
    
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    
    return X_train, y_train, X_test, y_test

def get_models_and_params():
    """Defines the models and their hyperparameter distributions."""
    models = {
        "Logistic Regression": (
            LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced'),
            {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear']}
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
            {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        ),
        "XGBoost": (
            XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss'),
            {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5], 'scale_pos_weight': [1, 3]}
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.8, 1.0]
            }
        )
    }
    return models

def train_and_optimize(X_train, y_train, X_test, y_test):
    """
    Runs RandomizedSearchCV for each model.
    Returns:
        best_overall_model: The single best model object.
        all_results: Dictionary containing all optimized models for visualization.
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
            n_iter=5,  # Reduced for speed
            scoring='roc_auc', # Optimizing for Discrimination not Accuracy
            cv=3,
            verbose=0,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        search.fit(X_train, y_train)
        best_estimator = search.best_estimator_
        
        # Evaluate
        y_pred = best_estimator.predict(X_test)
        
        # Get probabilities
        if hasattr(best_estimator, "predict_proba"):
            y_prob = best_estimator.predict_proba(X_test)[:, 1]
        else:
            y_prob = best_estimator.decision_function(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        
        # Store comprehensive results for visualization
        results[name] = {
            "accuracy": acc, 
            "roc_auc": roc, 
            "model": best_estimator,
            "best_params": search.best_params_,
            "y_prob": y_prob,     # Required for ROC/PR curves
            "y_pred": y_pred      # Required for Confusion Matrix
        }
        
        # Combined Score for Selection (Equal weight to Accuracy and ROC-AUC)
        combined_score = (roc + acc) / 2
        
        print(f"{name:<25} | {roc:.4f}     | {acc:.4f}     | Optimized (Score: {combined_score:.4f})")
        
        if combined_score > best_overall_score:
            best_overall_score = combined_score
            best_overall_model_name = name
            best_overall_model = best_estimator
            
    print("-" * 65)
    print(f"Best Model Selected: {best_overall_model_name} with Combined Score: {best_overall_score:.4f}")
    
    return best_overall_model, results

def visualize_performance(results, y_test):
    
    model_names = list(results.keys())
    n_models = len(model_names)
    

    rows = 1
    cols = n_models
    fig_size = (5 * cols, 5)
    
    # --- FIGURE 1: ROC-AUC Curves ---
    fig1, axes1 = plt.subplots(rows, cols, figsize=fig_size, constrained_layout=True)
    fig1.suptitle('ROC-AUC Curves Comparison', fontsize=16)
    
    if n_models == 1: axes1 = [axes1]
    
    for ax, name in zip(axes1, model_names):
        y_prob = results[name]['y_prob']
        roc_score = results[name]['roc_auc']
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_score:.2f}')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{name}')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    # --- FIGURE 2: Precision-Recall Curves ---
    fig2, axes2 = plt.subplots(rows, cols, figsize=fig_size, constrained_layout=True)
    fig2.suptitle('Precision-Recall Curves Comparison', fontsize=16)
    
    if n_models == 1: axes2 = [axes2]
    
    for ax, name in zip(axes2, model_names):
        y_prob = results[name]['y_prob']
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        avg_precision = average_precision_score(y_test, y_prob)
        
        ax.plot(recall, precision, color='green', lw=2, label=f'AP = {avg_precision:.2f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'{name}')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

    # --- FIGURE 3: Confusion Matrices ---
    fig3, axes3 = plt.subplots(rows, cols, figsize=fig_size, constrained_layout=True)
    fig3.suptitle('Confusion Matrices Comparison', fontsize=16)
    
    if n_models == 1: axes3 = [axes3]
    
    for ax, name in zip(axes3, model_names):
        y_pred = results[name]['y_pred']
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(f'{name}')
        ax.grid(False) 

    # Save Figure 3 (Confusion Matrix)
    if not os.path.exists(VISUALIZATIONS_DIR):
        os.makedirs(VISUALIZATIONS_DIR)
        
    fig3.savefig(os.path.join(VISUALIZATIONS_DIR, 'Figure_3.png'))
    plt.close(fig3)
    
    # Also save Figure 1 and 2 here for clarity (since they were created above)
    fig1.savefig(os.path.join(VISUALIZATIONS_DIR, 'Figure_1.png'))
    plt.close(fig1)
    
    fig2.savefig(os.path.join(VISUALIZATIONS_DIR, 'Figure_2.png'))
    plt.close(fig2)
    
    print(f"Visualizations saved to {VISUALIZATIONS_DIR}/")


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
        
        best_model, results = train_and_optimize(X_train, y_train, X_test, y_test)
        
        print("\n--- 3. Visualizing Results ---")
        visualize_performance(results, y_test)
        
        print("\n--- 4. Saving Best Model ---")
        save_model(best_model)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()