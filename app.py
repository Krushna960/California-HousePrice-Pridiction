"""
Flask app for Gurgaon House Price Prediction.
Serves the UI and prediction API using the trained model and pipeline.
"""
import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"
model = None
preprocessing_pipeline = None


def load_model():
    """Load model and pipeline if they exist."""
    global model, preprocessing_pipeline
    if os.path.exists(MODEL_FILE) and os.path.exists(PIPELINE_FILE):
        model = joblib.load(MODEL_FILE)
        preprocessing_pipeline = joblib.load(PIPELINE_FILE)
        return True
    return False


# Expected feature columns (same order as training)
FEATURE_COLS = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
    "ocean_proximity",
]


@app.route("/")
def index():
    """Serve the home/landing page."""
    return render_template("index.html", model_loaded=load_model())


@app.route("/predict")
def predict_page():
    """Serve the prediction form page."""
    return render_template("predict.html", model_loaded=load_model())


@app.route("/about")
def about():
    """Serve the about/help page."""
    return render_template("about.html", model_loaded=load_model())


@app.route("/api/predict", methods=["POST"])
def predict():
    """Accept form or JSON input and return predicted house value."""
    if model is None or preprocessing_pipeline is None:
        if not load_model():
            return jsonify({"error": "Model not found. Train the model first by running main.py."}), 503

    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form

        row = {}
        for col in FEATURE_COLS:
            val = data.get(col)
            if val is None or val == "":
                return jsonify({"error": f"Missing field: {col}"}), 400
            if col == "ocean_proximity":
                row[col] = str(val).strip().upper()
            else:
                try:
                    row[col] = float(val)
                except ValueError:
                    return jsonify({"error": f"Invalid number for {col}"}), 400

        df = pd.DataFrame([row])
        df = df[FEATURE_COLS]
        X = preprocessing_pipeline.transform(df)
        prediction = model.predict(X)[0]
        # Ensure non-negative and sensible format
        price = max(0, float(prediction))

        if request.is_json:
            return jsonify({"median_house_value": price, "prediction_inr": round(price * 83, 0)})
        return render_template(
            "predict.html",
            model_loaded=True,
            prediction=round(price, 2),
            prediction_inr=round(price * 83, 0),
            form_data=row,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_model()
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
