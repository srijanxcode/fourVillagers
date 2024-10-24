# Import necessary libraries
import pandas as pd
import mysql.connector
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Connect to MySQL database
def get_connection():
    return mysql.connector.connect(
        host='localhost',  # Your MySQL server host
        user='root',  # Your MySQL username
        password='mysqlyug',  # Your MySQL password
        database='h4hack'  # Your MySQL database name
    )

# Load the dataset from MySQL
def load_data():
    conn = get_connection()
    query = "SELECT * FROM packaged_food_items"
    data = pd.read_sql(query, conn)
    conn.close()
    return data

# Load the data
data = load_data()

# Encode categorical variables
label_encoders = {}
for column in ['main_ingredient', 'nutritional_information', 'packaging_type']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Prepare the features (X) and target (y)
X = data[['main_ingredient', 'shelf_life', 'nutritional_information', 'packaging_type']]
y = data['compliance_status']

# Encode target variable
compliance_encoder = LabelEncoder()
y = compliance_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Display valid options for user input from the MySQL table
valid_main_ingredients = list(label_encoders['main_ingredient'].classes_)
valid_nutritional_info = list(label_encoders['nutritional_information'].classes_)
valid_packaging_types = list(label_encoders['packaging_type'].classes_)

# Streamlit user interface
st.title("Food Compliance Prediction")

main_ingredient = st.selectbox("Select the Main Ingredient", valid_main_ingredients)
shelf_life = st.number_input("Enter the Shelf Life (in days)", min_value=1, step=1)
nutritional_info = st.selectbox("Select the Nutritional Information", valid_nutritional_info)
packaging_type = st.selectbox("Select the Packaging Type", valid_packaging_types)

if st.button("Predict Compliance"):
    # Encode user input using the same LabelEncoders
    main_ingredient_encoded = label_encoders['main_ingredient'].transform([main_ingredient])[0]
    nutritional_info_encoded = label_encoders['nutritional_information'].transform([nutritional_info])[0]
    packaging_type_encoded = label_encoders['packaging_type'].transform([packaging_type])[0]
    
    # Create a feature vector for prediction
    features = np.array([[main_ingredient_encoded, shelf_life, nutritional_info_encoded, packaging_type_encoded]])
    
    # Predict and decode the result
    prediction = model.predict(features)
    compliance_status = compliance_encoder.inverse_transform(prediction)
    
    st.write(f"The food item is predicted to be: **{compliance_status[0]}**")
