import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/fraud_detection_model.pkl")

st.set_page_config(page_title="Credit Card Fraud Detection")

st.title("💳 Credit Card Fraud Detection")
st.write("Upload a CSV file to detect fraudulent transactions.")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:

    # Read CSV
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.write(df.head())

    # Remove Class column if it exists
    if "Class" in df.columns:
        df = df.drop("Class", axis=1)

    if st.button("Predict Fraud"):

        predictions = model.predict(df)

        result = df.copy()
        result["Prediction"] = predictions

        st.subheader("Prediction Results")
        st.write(result.head())

        fraud = (predictions == 1).sum()
        legit = (predictions == 0).sum()

        st.success(f"✅ Legitimate Transactions : {legit}")
        st.error(f"🚨 Fraudulent Transactions : {fraud}")

        csv = result.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Results",
            data=csv,
            file_name="prediction_results.csv",
            mime="text/csv"
        )