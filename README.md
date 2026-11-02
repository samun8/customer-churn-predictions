# Customer Churn Prediction Project

This project aims to predict telecom customer churn using machine learning. A logistic regression model is built, trained, and saved to classify whether a customer is likely to leave (churn) or stay with the company.

## Project Description
Customer churn is a critical metric for subscription-based businesses. By identifying customers at risk of churn, businesses can proactively offer incentives or improve services to retain them. This project uses the Telco Customer Churn dataset from Kaggle to build an end-to-end predictive pipeline.

## Dependencies List
Ensure you have the following Python packages installed:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `joblib`
- `kagglehub`

## Installation Instructions
To set up and run this project, install the required packages:
```bash
pip install kagglehub pandas numpy matplotlib seaborn scikit-learn joblib
```

## Usage Examples
### 1. Training and Saving the Model
The model training pipeline automatically processes numerical and categorical columns before feeding them into a Logistic Regression classifier:
```python
import joblib
# Save the model pipeline
joblib.dump(model_pipeline, 'customer_churn_model.pkl')
```

### 2. Loading and Predicting on New Data
To make predictions on new client profiles:
```python
import pandas as pd
import joblib

# Load the saved model pipeline
model = joblib.load('customer_churn_model.pkl')

# Create a new customer profile
new_customer = pd.DataFrame({
    'gender': ['Male'],
    'SeniorCitizen': [0],
    'Partner': ['No'],
    'Dependents': ['No'],
    'tenure': [1],
    'PhoneService': ['Yes'],
    'MultipleLines': ['No'],
    'InternetService': ['Fiber optic'],
    'OnlineSecurity': ['No'],
    'OnlineBackup': ['No'],
    'DeviceProtection': ['No'],
    'TechSupport': ['No'],
    'StreamingTV': ['No'],
    'StreamingMovies': ['No'],
    'Contract': ['Month-to-month'],
    'PaperlessBilling': ['Yes'],
    'PaymentMethod': ['Electronic check'],
    'MonthlyCharges': [75.75],
    'TotalCharges': [75.75]
})

prediction = model.predict(new_customer)
print("Predicted Churn:", 'Yes' if prediction[0] == 1 else 'No')
```

## Key Findings Summary
- **Model Performance**: The Logistic Regression model achieved an accuracy of approximately **80.38%** on the hold-out test set.
- **Top Churn Drivers**:
  1. **Tenure**: Long-term customers are significantly less likely to churn.
  2. **Contract Type**: Month-to-month contract holders have the highest propensity to churn, whereas Two-Year contracts strongly discourage churn.
  3. **Internet Service**: Customers using Fiber Optic internet show higher rates of churn compared to DSL or those with no internet service.
