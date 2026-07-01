import random
import time

import requests

url = "http://localhost:8000/predict"


def generate_toxic_customer():
    return {
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 118.75,
        "TotalCharges": 118.75,
    }


def generate_loyal_customer():
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 72,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "No",
        "OnlineSecurity": "No internet service",
        "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service",
        "TechSupport": "No internet service",
        "StreamingTV": "No internet service",
        "StreamingMovies": "No internet service",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 20.0,
        "TotalCharges": 1400.0,
    }


print("Starting EXTREME Load Test...")

while True:
    if random.random() > 0.5:
        data = generate_toxic_customer()
        type_cust = "TOXIC"
    else:
        data = generate_loyal_customer()
        type_cust = "LOYAL"

    try:
        response = requests.post(url, json=data)

        if response.status_code == 200:
            res = response.json()
            pred = res["message"]
            prob = res["probability"]

            print(f"Sent: {type_cust} | Model: {prob:.4f} -> {pred}")
        else:
            print(f"Error {response.status_code}: {response.text}")

    except Exception as e:
        print(f"Connection error: {e}")

    time.sleep(0.2)
