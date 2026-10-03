import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize the Flask application
superkart_revenue_predict_api = Flask("SuperKart revenue Predictor")

# Load the trained model (trained on one-hot encoded data, see the notebook)
model = joblib.load("superkart_model.joblib")
MODEL_COLUMNS = list(model.feature_names_in_)

# Same categorical columns that were one-hot encoded during training
CATEGORICAL_COLS = [
    "Product_Type",
    "Product_Sugar_Content",
    "Store_Location_City_Type",
    "Store_Type",
    "Store_Size",
]

REQUIRED_FIELDS = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Establishment_Year",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Type",
]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same encoding as training and align to the model's columns."""
    df = df.drop(
        columns=[c for c in ("Product_Id", "Store_Id", "Product_Store_Sales_Total") if c in df.columns]
    )
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS)
    # Adds any missing dummy columns as 0 and drops/reorders the rest
    df = df.reindex(columns=MODEL_COLUMNS, fill_value=0)
    return df.astype(float)


@superkart_revenue_predict_api.get("/")
def home():
    return "Welcome to the SuperKart store wise product revenue Prediction API!"


@superkart_revenue_predict_api.post("/v1/predict")
def predict_revenue():
    data = request.get_json(silent=True) or {}

    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    input_df = prepare_features(pd.DataFrame([{f: data[f] for f in REQUIRED_FIELDS}]))
    prediction = float(model.predict(input_df)[0])  # target was not log-transformed

    return jsonify({"Predicted revenue (in dollars)": round(prediction, 2)})


@superkart_revenue_predict_api.post("/v1/predictbatch")
def predict_revenue_batch():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded under key 'file'"}), 400

    raw = pd.read_csv(request.files["file"])
    predictions = model.predict(prepare_features(raw.copy()))

    # Product_Id repeats across stores, so combine with Store_Id when available
    if "Product_Id" in raw.columns and "Store_Id" in raw.columns:
        keys = (raw["Product_Id"].astype(str) + "_" + raw["Store_Id"].astype(str)).tolist()
    elif "Product_Id" in raw.columns:
        keys = raw["Product_Id"].astype(str).tolist()
    else:
        keys = [str(i) for i in range(len(raw))]

    return jsonify({k: round(float(p), 2) for k, p in zip(keys, predictions)})


if __name__ == "__main__":
    superkart_revenue_predict_api.run(host="0.0.0.0", port=7860)
