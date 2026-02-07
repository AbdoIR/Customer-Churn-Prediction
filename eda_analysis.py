"""
Exploratory Data Analysis (EDA) Script for Customer Churn Prediction.

This script performs comprehensive EDA on the Telco Customer Churn dataset, including:
1. Data Loading and Cleaning.
2. Target Variable Analysis (Class Imbalance).
3. Numerical Feature Analysis (Distribution, Outliers).
4. Categorical Feature Analysis (Churn Rates by Category).
5. Correlation Analysis.

Outputs:
- Statistical summaries to console.
- Plots saved to 'eda_results/' directory.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
DATA_PATH = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
OUTPUT_DIR = 'eda_results'

def setup_output_dir():
    """Creates the output directory if it does not exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

def load_data(filepath):
    """
    Loads and cleans the dataset.
    - Handles 'TotalCharges' which contains empty strings.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
        
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Cleaning: TotalCharges is object, needs to be numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Drop rows with missing TotalCharges (very few)
    initial_len = len(df)
    df.dropna(subset=['TotalCharges'], inplace=True)
    print(f"Data loaded. Dropped {initial_len - len(df)} rows with missing charges.")
    
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    return df

def analyze_target(df, target='Churn'):
    """Analyzes and plots the distribution of the target variable."""
    print(f"\n--- Target Analysis ({target}) ---")
    print(df[target].value_counts(normalize=True))
    
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target, data=df, palette='viridis')
    plt.title(f'Distribution of {target}')
    plt.savefig(os.path.join(OUTPUT_DIR, 'target_distribution.png'))
    plt.close()

def analyze_numerical(df, numerical_cols, target=None):
    """
    Analyzes numerical features: descriptive stats, distributions, and boxplots versus target.
    """
    print("\n--- Numerical Feature Analysis ---")
    print(df[numerical_cols].describe())
    
    for col in numerical_cols:
        print(f"\nAnalyzing {col}...")
        
        # Distribution
        plt.figure(figsize=(8, 4))
        sns.histplot(data=df, x=col, hue=target, kde=True, element="step", palette='viridis')
        plt.title(f'Distribution of {col} by {target}')
        plt.savefig(os.path.join(OUTPUT_DIR, f'dist_{col}.png'))
        plt.close()
        
        # Boxplot
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=target, y=col, data=df, palette='viridis')
        plt.title(f'Boxplot of {col} by {target}')
        plt.savefig(os.path.join(OUTPUT_DIR, f'boxplot_{col}.png'))
        plt.close()

def analyze_categorical(df, categorical_cols, target='Churn'):
    """
    Analyzes categorical features showing churn rates for each category.
    """
    print("\n--- Categorical Feature Analysis ---")
    for col in categorical_cols:
        if df[col].nunique() > 50:
            continue # Skip high cardinality
            
        # Plot
        plt.figure(figsize=(10, 6))
        sns.countplot(x=col, hue=target, data=df, palette='rocket')
        plt.title(f'Churn by {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'cat_{col}.png'))
        plt.close()

def analyze_correlations(df, numerical_cols):
    """Plots correlation heatmap for numerical features."""
    print("\n--- Correlation Analysis ---")
    corr = df[numerical_cols].corr()
    print(corr)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix (Numerical)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'))
    plt.close()

def main():
    setup_output_dir()
    df = load_data(DATA_PATH)
    
    target = 'Churn'
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_cols = [c for c in df.columns if c not in numerical_cols and c != target]
    
    analyze_target(df, target)
    analyze_numerical(df, numerical_cols, target)
    analyze_categorical(df, categorical_cols, target)
    analyze_correlations(df, numerical_cols)
    
    print(f"\nEDA Complete. Results saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
