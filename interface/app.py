import sys
import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

# Add project root to path to import preprocessing_pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from preprocessing_pipeline
from preprocessing_pipeline import TelcoFeatureEngineer

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Models
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl')
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'preprocessor.pkl')

model = None
preprocessor = None

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print("Models loaded successfully.")
    else:
        print(f"Models not found at {MODEL_PATH} or {PREPROCESSOR_PATH}")
except Exception as e:
    print(f"Error loading models: {e}")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_data(filepath):
    # 1. Load Data
    df = pd.read_csv(filepath)
    
    # Preserve original data with customerID
    original_data = df.copy()
    
    # 2. Cleaning (Mirroring load_and_clean_data but keeping IDs in 'data' separate)
    if 'customerID' in df.columns:
        work_df = df.drop(columns=['customerID'])
    else:
        work_df = df.copy()
        
    work_df['TotalCharges'] = pd.to_numeric(work_df['TotalCharges'], errors='coerce')
    work_df['TotalCharges'] = work_df['TotalCharges'].fillna(0)
    work_df['TotalCharges'] = np.log1p(work_df['TotalCharges'])
    
    # 3. Feature Engineering
    fe = TelcoFeatureEngineer()
    # Note: 'fit' does nothing, 'transform' adds columns
    work_df_eng = fe.transform(work_df)
    
    # Drop TotalCharges as per pipeline
    if 'TotalCharges' in work_df_eng.columns:
        work_df_eng.drop(columns=['TotalCharges'], inplace=True)
        
    # 4. Preprocessing (Scaling/Encoding)
    if preprocessor:
        X_processed = preprocessor.transform(work_df_eng)
        
        # 5. Prediction
        if model:
            predictions = model.predict(X_processed)
            # Check if model supports proba
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_processed)[:, 1]
            else:
                probabilities = [0.0] * len(predictions)
                
            original_data['Churn_Prediction'] = predictions
            original_data['Churn_Probability'] = probabilities
            original_data['Churn_Prediction_Label'] = original_data['Churn_Prediction'].map({1: 'Yes', 0: 'No'})
            
    return original_data

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                results = process_data(filepath)
                
                # Stats
                total = len(results)
                churned = int(results['Churn_Prediction'].sum())
                not_churned = total - churned
                churn_rate = round((churned / total) * 100, 1) if total > 0 else 0
                
                # Churn by Contract Type
                contract_churn = {}
                if 'Contract' in results.columns:
                    results_clean = results.dropna(subset=['Contract'])
                    contract_data = results_clean.groupby('Contract')['Churn_Prediction'].agg(['sum', 'count'])
                    contract_churn = {str(idx): {'churned': int(row['sum']), 'total': int(row['count'])} 
                                     for idx, row in contract_data.iterrows()}
                
                # Churn by Tenure Groups
                tenure_churn = {}
                if 'tenure' in results.columns:
                    results['TenureGroup'] = pd.cut(results['tenure'], 
                                                     bins=[0, 12, 24, 36, 48, 100], 
                                                     labels=['0-12', '13-24', '25-36', '37-48', '49+'])
                    results_clean = results.dropna(subset=['TenureGroup'])
                    tenure_data = results_clean.groupby('TenureGroup', observed=True)['Churn_Prediction'].agg(['sum', 'count'])
                    tenure_churn = {str(idx): {'churned': int(row['sum']), 'total': int(row['count'])} 
                                   for idx, row in tenure_data.iterrows()}
                
                # Churn by Internet Service
                internet_churn = {}
                if 'InternetService' in results.columns:
                    results_clean = results.dropna(subset=['InternetService'])
                    internet_data = results_clean.groupby('InternetService')['Churn_Prediction'].agg(['sum', 'count'])
                    internet_churn = {str(idx): {'churned': int(row['sum']), 'total': int(row['count'])} 
                                     for idx, row in internet_data.iterrows()}
                
                # Monthly Charges Distribution
                high_risk = results[results['Churn_Prediction'] == 1]
                low_risk = results[results['Churn_Prediction'] == 0]
                
                monthly_charges_churned = high_risk['MonthlyCharges'].tolist() if 'MonthlyCharges' in high_risk.columns else []
                monthly_charges_retained = low_risk['MonthlyCharges'].tolist() if 'MonthlyCharges' in low_risk.columns else []
                
                # Risk Distribution (Probability bins)
                risk_bins = pd.cut(results['Churn_Probability'], 
                                  bins=[0, 0.3, 0.5, 0.7, 1.0], 
                                  labels=['Low', 'Medium', 'High', 'Critical'])
                risk_distribution = risk_bins.value_counts().to_dict()
                risk_distribution = {str(k): int(v) for k, v in risk_distribution.items() if pd.notna(k)}
                
                # Top 10 at risk customers
                if 'customerID' in results.columns:
                    top_risk_cols = ['customerID', 'Churn_Probability', 'MonthlyCharges', 'tenure', 'Contract']
                    top_risk_cols = [col for col in top_risk_cols if col in results.columns]
                    top_risk = results.nlargest(10, 'Churn_Probability')[top_risk_cols]
                else:
                    # If no customerID, use index and show other relevant columns
                    top_risk_cols = ['Churn_Probability', 'MonthlyCharges', 'tenure', 'Contract', 'InternetService']
                    top_risk_cols = [col for col in top_risk_cols if col in results.columns]
                    top_risk = results.nlargest(10, 'Churn_Probability')[top_risk_cols]
                
                # Format probability as percentage
                if 'Churn_Probability' in top_risk.columns:
                    top_risk['Churn_Probability'] = (top_risk['Churn_Probability'] * 100).round(1).astype(str) + '%'

                return render_template('dashboard.html', 
                                       total=total,
                                       churned=churned,
                                       not_churned=not_churned,
                                       churn_rate=churn_rate,
                                       contract_churn=contract_churn,
                                       tenure_churn=tenure_churn,
                                       internet_churn=internet_churn,
                                       monthly_charges_churned=monthly_charges_churned,
                                       monthly_charges_retained=monthly_charges_retained,
                                       risk_distribution=risk_distribution,
                                       top_risk_table=top_risk.to_html(classes='table table-sm table-hover', header="true", index=False),
                                       tables=[results.head(20).to_html(classes='table table-striped table-hover', header="true", index=False)])
                                       
            except Exception as e:
                flash(f'Error processing file: {e}')
                print(e)
                return redirect(request.url)
                
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
