from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Real-Time Fraud Detection API", version="1.0.0")

# Prometheus Metrics Instrumentation
Instrumentator().instrument(app).expose(app)

# Trained Random Forest Model load karein
model = joblib.load("fraud_model.pkl")

class TransactionData(BaseModel):
    amount: float
    oldBalance: float
    newBalance: float

@app.get("/")
def home():
    return {"status": "Online", "message": "Fraud Detection API is Running"}

@app.post("/predict")
def predict_fraud(data: TransactionData):
    features = np.array([[data.amount, data.oldBalance, data.newBalance]])
    prediction = model.predict(features)[0]
    result = "Fraudulent" if prediction == 1 else "Legitimate"
    return {"prediction": result, "status_code": 200}