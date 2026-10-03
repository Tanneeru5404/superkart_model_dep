# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
superkart_revenue_predict_api = Flask("SuperKart revenue Predictor")

# Load the trained machine learning model
model = joblib.load("superkart_model.joblib")

# Define a route for the home page (GET request)
@superkart_revenue_predict_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the SuperKart store wise prodcut revenue Prediction API!"

# Define an endpoint for single property prediction (POST request)
@superkart_revenue_predict_api.post('/v1/predict')
def predict_revenue_price():
    """
    This function handles POST requests to the '/v1/rental' endpoint.
    It expects a JSON payload containing property details and returns
    the predicted rental price as a JSON response.
    """
    # Get the JSON data from the request body
    property_data = request.get_json()

    # Extract relevant features from the JSON data
    sample = {
        'Product_Weight': property_data['Product_Weight'],
        'Product_Sugar_Content': property_data['Product_Sugar_Content'],
        'Product_Allocated_Area': property_data['Product_Allocated_Area'],
        'Product_MRP': property_data['Product_MRP'],
        'Store_Size': property_data['Store_Size'],
        'Store_Location_City_Type': property_data['Store_Location_City_Type'],
        'Store_Type': property_data['Store_Type'],
        'Store_Establishment_Year': property_data['Store_Establishment_Year'],
        'Product_Type': property_data['Product_Type']
    }

    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([sample])

    # Make prediction (get Product_Store_Sales_Total)
    predicted_Product_Store_Sales_Total = model.predict(input_data)[0]

    # Calculate actual price
    predicted_price = np.exp(predicted_Product_Store_Sales_Total)

    # Convert predicted_price to Python float
    predicted_price = round(float(predicted_price), 2)
    # The conversion above is needed as we convert the model prediction (log price) to actual price using np.exp, which returns predictions as NumPy float32 values.
    # When we send this value directly within a JSON response, Flask's jsonify function encounters a datatype error

    # Return the actual price
    return jsonify({'Predicted revenue (in dollars)': predicted_price})


# Define an endpoint for batch prediction (POST request)
@superkart_revenue_predict_api.post('/v1/predictbatch')
def predict_revenue_batch():
    """
    This function handles POST requests to the '/v1/rentalbatch' endpoint.
    It expects a CSV file containing property details for multiple properties
    and returns the predicted rental prices as a dictionary in the JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Make predictions for all properties in the DataFrame (get log_prices)
    predicted_log_prices = model.predict(input_data).tolist()

    # Calculate actual prices
    predicted_prices = [round(float(np.exp(Product_Store_Sales_Total)), 2) for Product_Store_Sales_Total in predicted_log_prices]

    # Create a dictionary of predictions with property IDs as keys
    property_ids = input_data['id'].tolist()  # Assuming 'id' is the property ID column
    output_dict = dict(zip(property_ids, predicted_prices))  # Use actual prices

    # Return the predictions dictionary as a JSON response
    return output_dict

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    superkart_revenue_predict_api.run(debug=True)
