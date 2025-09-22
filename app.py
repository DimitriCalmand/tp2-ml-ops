from flask import app
import mlflow.pyfunc
import pandas as pd
from mlflow.tracking import MlflowClient
from flask import Flask, request, jsonify
import os

client = MlflowClient()

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

REGISTERED_MODEL_NAME = "sk-learn-random-forest-reg-model"

def load_model(version="latest"):
    try:
        # Try to load from registered model first
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/latest"
        model = mlflow.sklearn.load_model(model_uri)
        return model
    except Exception as e:
        print(f"Could not load registered model: {e}")
        # Fallback to loading directly from model artifacts
        try:
            # Direct path to the models
            model_paths = [
                "/app/mlruns/729775294814508479/models/m-16db3a829308407d8db1d986d21b4533/artifacts",
                "/app/mlruns/729775294814508479/models/m-fbc262b3cda241908025c4963da8d70b/artifacts"
            ]
            
            for model_path in model_paths:
                if os.path.exists(model_path):
                    model = mlflow.sklearn.load_model(model_path)
                    print(f"Loaded model from {model_path}")
                    return model
            
            raise Exception("No model found in any of the expected paths")
        except Exception as fallback_error:
            print(f"Fallback loading failed: {fallback_error}")
            raise Exception(f"Failed to load model: {e}")

model = load_model(version="latest")

app = Flask(__name__)

#helloworld on / rout
@app.route("/", methods=["GET"])
def hello_world():
    return "Hello, World!"

@app.route("/predict", methods=["POST"])
def predict():
    global model
    try:
        json_data = request.get_json()
        features = json_data.get("features", [])
        data = pd.DataFrame(features)
        preds = model.predict(data)
        return jsonify({"predictions": preds.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/update-model", methods=["POST"])
def update_model():
    global model
    try:
        json_data = request.get_json()
        version = json_data.get("version", None)

        # Vérifier que la version/stage existe
        if version:
            versions = [mv.version for mv in client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")]
            if version not in [int(v) for v in versions]:
                return jsonify({"error": f"Version {version} does not exist."}), 400

        # Charger le modèle
        model = load_model(version=version)
        return jsonify({"message": f"Model updated successfully (version={version})"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)