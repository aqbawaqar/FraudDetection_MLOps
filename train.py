import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# MLflow Experiment setup
mlflow.set_experiment("Fraud_Detection_Experiment")

# Synthetic Dataset Generation
np.random.seed(42)
n_samples = 1000
amount = np.random.uniform(10, 50000, n_samples)
oldBalance = np.random.uniform(100, 100000, n_samples)
newBalance = oldBalance - amount
newBalance[newBalance < 0] = 0
isFraud = np.where((amount > 20000) & (newBalance == 0), 1, 0)

df = pd.DataFrame({
    'amount': amount, 
    'oldBalance': oldBalance, 
    'newBalance': newBalance, 
    'isFraud': isFraud
})

X = df[['amount', 'oldBalance', 'newBalance']]
y = df['isFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow Tracking Run
with mlflow.start_run():
    n_estimators = 100
    max_depth = 10
    
    # Random Forest Model
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    
    accuracy = clf.score(X_test, y_test)
    
    # MLflow Params, Metrics & Model Logging
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)
    
    # Updated log_model with skops trusted types fix
    mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="random_forest_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )
    
    # Save model locally for FastAPI inference
    joblib.dump(clf, "fraud_model.pkl")
    print(f"Model trained successfully. Accuracy: {accuracy * 100:.2f}%")