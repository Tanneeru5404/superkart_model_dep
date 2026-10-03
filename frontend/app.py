import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend
BACKEND_URL = "http://backend:7860"

# Set the title of the Streamlit app
st.title("SuperKart Store-wise Product Revenue Prediction")

# Section for online prediction
st.subheader("Online Prediction")

# Collect user input based on SuperKart dataset attributes
Product_Weight = st.number_input("Product Weight (in kg)", min_value=4.0, max_value=30.0, step=0.1, value=12.6)
Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
Product_Allocated_Area = st.number_input("Product Allocated Area Ratio", min_value=0.0, max_value=1.0, step=0.01, value=0.05)
Product_MRP = st.number_input("Product Maximum Retail Price (MRP)", min_value=10.0, max_value=500.0, step=1.0, value=147.0)
Store_Establishment_Year = st.number_input("Store Establishment Year", min_value=1980, max_value=2025, step=1, value=2009)
Store_Size = st.selectbox("Store Size", ["Medium", "High", "Small"])
Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])
Product_Type = st.selectbox("Product Type Category", [
    "Baking Goods", "Breads", "Breakfast", "Canned", "Dairy", "Frozen Foods",
    "Fruits and Vegetables", "Hard Drinks", "Health and Hygiene", "Household",
    "Meat", "Others", "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods"
])

# Convert user input into a JSON structure matching the preprocess pipeline
# We will reconstruct the feature columns exactly as expected by the backend model.
payload = {
    'Product_Weight': Product_Weight,
    'Product_Sugar_Content': Product_Sugar_Content,
    'Product_Allocated_Area': Product_Allocated_Area,
    'Product_MRP': Product_MRP,
    'Store_Establishment_Year': Store_Establishment_Year,
    'Store_Size': Store_Size,
    'Store_Location_City_Type': Store_Location_City_Type,
    'Store_Type': Store_Type,
    'Product_Type': Product_Type
}

# Make prediction when the "Predict" button is clicked
if st.button("Predict Revenue", type="primary"):
    response = requests.post(f"{BACKEND_URL}/v1/rental", json=payload)  # Send payload to Flask API
    if response.status_code == 200:
        try:
            prediction = response.json()['Predicted revenue (in dollars)']
            st.success(f"Predicted Product Store Sales Total: ${prediction:,.2f}")
        except KeyError:
            st.error(f"Response format error. API returned: {response.text}")
    else:
        st.error(f"Unable to connect to the prediction API. Status code: {response.status_code}")

# Section for batch prediction
st.subheader("Batch Prediction")

# Allow users to upload a CSV file for batch prediction
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

# Make batch prediction when the "Predict Batch" button is clicked
if uploaded_file is not None:
    if st.button("Predict Batch", type="primary"):
        response = requests.post(f"{BACKEND_URL}/v1/rentalbatch", files={"file": uploaded_file})  # Send file to Flask API
        if response.status_code == 200:
            predictions = response.json()
            st.success("Batch predictions completed!")
            st.write(predictions)  # Display the predictions
        else:
            st.error("Unable to connect to the prediction API.")
