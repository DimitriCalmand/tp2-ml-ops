import requests
import json
import numpy as np

# URL de ton API Flask
BASE_URL = "http://127.0.0.1:8000"

# --- 1️⃣ Test /predict ---
def test_predict():
    # Exemple : 2 observations avec 10 features
    features = np.random.rand(2, 10).tolist()
    payload = {"features": features}
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    
    if response.status_code == 200:
        print("✅ /predict response:", response.json())
    else:
        print("❌ /predict error:", response.status_code, response.text)

# --- 2️⃣ Test /update-model ---
def test_update_model(stage=None, version=None):
    payload = {}
    if stage:
        payload["stage"] = stage
    if version:
        payload["version"] = version

    response = requests.post(f"{BASE_URL}/update-model", json=payload)
    
    if response.status_code == 200:
        print(f"✅ /update-model response:", response.json())
    else:
        print(f"❌ /update-model error:", response.status_code, response.text)

# --- Exécution ---
if __name__ == "__main__":
    print("Testing /predict...")
    test_predict()
    
    print("\nTesting /update-model to Staging...")
    test_update_model(version=2)
    
    print("\nTesting /update-model to version 1...")
    test_update_model(version=1)
