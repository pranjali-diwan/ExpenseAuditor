import streamlit as st
import pandas as pd
import joblib
import os

MODELS_DIR = os.path.join("..", "models")
PRIMARY_MODEL_PATH = os.path.join(MODELS_DIR, "expense_auditor.pkl")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "expense_auditor_rf.joblib")
IFOREST_MODEL_PATH = os.path.join(MODELS_DIR, "expense_auditor_iforest.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(MODELS_DIR, "encoders.pkl")

@st.cache_resource
def load_assets():
    if not os.path.isdir(MODELS_DIR):
        raise FileNotFoundError(f"Models directory not found at {MODELS_DIR}")

    loaded_model = None
    loaded_scaler = None
    loaded_encoders = None

    # Try consolidated artifact first
    consolidated_path = None
    if os.path.exists(PRIMARY_MODEL_PATH):
        consolidated_path = PRIMARY_MODEL_PATH
    
    if consolidated_path is not None:
        obj = joblib.load(consolidated_path)
        # Support dict artifacts or tuple/list
        if isinstance(obj, dict):
            loaded_model = obj.get("model", obj.get("estimator"))
            loaded_scaler = obj.get("scaler")
            loaded_encoders = obj.get("encoders")
        elif isinstance(obj, (list, tuple)):
            # take best-effort first three items
            if len(obj) >= 1:
                loaded_model = obj[0]
            if len(obj) >= 2:
                loaded_scaler = obj[1]
            if len(obj) >= 3:
                loaded_encoders = obj[2]
        else:
            # Single estimator only
            loaded_model = obj

    # If any pieces missing, try separate files
    if loaded_scaler is None and os.path.exists(SCALER_PATH):
        loaded_scaler = joblib.load(SCALER_PATH)
    if loaded_encoders is None and os.path.exists(ENCODERS_PATH):
        loaded_encoders = joblib.load(ENCODERS_PATH)
    if loaded_model is None:
        # fallback to alternative model artifacts
        if os.path.exists(RF_MODEL_PATH):
            loaded_model = joblib.load(RF_MODEL_PATH)
        elif os.path.exists(IFOREST_MODEL_PATH):
            loaded_model = joblib.load(IFOREST_MODEL_PATH)

    if loaded_model is None:
        raise FileNotFoundError("No model artifact found in the models directory.")

    return loaded_model, loaded_scaler, loaded_encoders


def preprocess_data(df, scaler, encoders):
    # Encode categorical columns safely (if encoders provided)
    if encoders is not None:
        for col, encoder in encoders.items():
            if col in df.columns:
                df[col] = df[col].map(lambda x: encoder.get(x, -1))  # unseen → -1
    # Scale numerical values (if scaler provided)
    if scaler is not None:
        numeric_cols = df.select_dtypes(include=["float64", "int64", "float32", "int32"]).columns
        if len(numeric_cols) > 0:
            df_scaled = scaler.transform(df[numeric_cols])
            df[numeric_cols] = df_scaled
    return df


def main():
    st.title("💰 Expense Auditor")

    model, scaler, encoders = load_assets()

    uploaded_file = st.file_uploader("Upload Expense CSV", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("📂 Uploaded Data:", df.head())

            # Preprocess
            processed_df = preprocess_data(df.copy(), scaler, encoders)

            # Predict
            preds = model.predict(processed_df)
            df["Prediction"] = preds

            st.success("✅ Predictions completed")
            st.write(df)

        except Exception as e:
            st.error(f"Error while predicting: {e}")


if __name__ == "__main__":
    main()
