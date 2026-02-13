# Customer Churn Prediction Web Interface

## Overview
A modern, AI-powered Flask web application that predicts customer churn using machine learning. Upload customer data and get instant predictions with comprehensive analytics.

## Features

### 🎨 Modern UI/UX
- Animated gradient backgrounds
- Smooth transitions and hover effects
- Responsive design for all devices
- Professional dashboard layout

### 📊 Comprehensive Analytics
The dashboard provides multiple visualizations to showcase model predictions:

1. **KPI Cards**: Total customers, churned, retained, and churn rate
2. **Churn Distribution**: Pie chart showing overall churn vs retention
3. **Contract Analysis**: How churn varies by contract type (Month-to-month, One year, Two year)
4. **Tenure Analysis**: Churn trends across customer tenure groups
5. **Internet Service Impact**: Churn patterns by internet service type
6. **Risk Distribution**: Customer segmentation by churn probability (Low, Medium, High, Critical)
7. **High-Risk Customers**: Table of top 10 customers most likely to churn

### 🤖 AI Insights
- Real-time predictions for each customer
- Churn probability scores (0-100%)
- Actionable insights highlighting business value

## Installation

1. Ensure all dependencies are installed:
```bash
conda activate deep_env
pip install -r requirements.txt
```

2. Make sure you have trained models in the `models/` directory:
   - `best_model.pkl` - Trained classification model
   - `preprocessor.pkl` - Data preprocessing pipeline

## Usage

1. Start the Flask application:
```bash
conda activate deep_env
python interface/app.py
```

2. Open your browser and navigate to:
```
http://127.0.0.1:5000
```

3. Upload your CSV file:
   - File must have the same columns as the training data
   - Supported format: Telco Customer Churn dataset structure
   - Required columns: tenure, MonthlyCharges, Contract, InternetService, etc.

4. View the dashboard:
   - Analyze churn predictions
   - Identify high-risk customers
   - Understand churn patterns

## File Structure
```
interface/
├── app.py                  # Flask application
├── templates/
│   ├── index.html         # Upload page
│   └── dashboard.html     # Analytics dashboard
├── static/                # Static assets (if needed)
├── uploads/               # Uploaded CSV files (created automatically)
└── README.md             # This file
```

## How It Works

1. **Data Upload**: User uploads CSV file with customer data
2. **Preprocessing**: 
   - Clean and transform data
   - Apply feature engineering (same as training pipeline)
   - Scale and encode features
3. **Prediction**: 
   - Feed processed data to trained model
   - Generate churn predictions and probabilities
4. **Visualization**: 
   - Aggregate results by various dimensions
   - Create interactive charts
   - Display insights and recommendations

## Model Value Demonstration

The interface clearly shows the model's value by:
- **Identifying at-risk customers** before they churn
- **Quantifying churn probability** for prioritization
- **Revealing patterns** (e.g., Month-to-month contracts have higher churn)
- **Enabling proactive retention** strategies
- **Potential cost savings** through early intervention

## Troubleshooting

**Issue**: ModuleNotFoundError
- **Solution**: Activate conda environment and install requirements

**Issue**: Model not found
- **Solution**: Run `model_training.py` first to generate model files

**Issue**: CSV format error
- **Solution**: Ensure CSV has same columns as training data

## Future Enhancements
- Export predictions to CSV
- Batch processing for large files
- Custom threshold settings
- Email alerts for high-risk customers
- Historical trend analysis
- A/B testing of retention strategies

## Technical Stack
- **Backend**: Flask (Python)
- **Frontend**: Bootstrap 5, Chart.js
- **ML**: Scikit-learn, XGBoost
- **Data**: Pandas, NumPy
