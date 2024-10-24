import pandas as pd
import mysql.connector
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Database connection
def get_connection():
    return mysql.connector.connect(
        host='localhost',  
        user='root',  
        password='mysqlyug',  
        database='h4hack'  
    )

# Load data from MySQL
def load_packaged_food_data():
    conn = get_connection()
    query = "SELECT * FROM packaged_food_items"
    data = pd.read_sql(query, conn)
    conn.close()
    return data

def load_vegetable_data():
    conn = get_connection()
    query = "SELECT * FROM vegetable_data"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Encode categorical columns for both datasets
def encode_columns(data, columns):
    label_encoders = {}
    for column in columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    return label_encoders

# Train model for packaged food compliance prediction
def train_packaged_food_model(data):
    X = data[['main_ingredient', 'shelf_life', 'nutritional_information', 'packaging_type']]
    y = data['compliance_status']
    
    compliance_encoder = LabelEncoder()
    y = compliance_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, compliance_encoder

# Train model for vegetable safety prediction
def train_vegetable_model(df):
    features = ['weather_conditions', 'pH_level', 'freshness_level', 'storage_value', 'pest_control_methods', 'nutrient_levels']
    target = 'safety_certification'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    return clf, features, target

# Load data
packaged_data = load_packaged_food_data()
vegetable_data = load_vegetable_data()

# Encode packaged food data
packaged_label_encoders = encode_columns(packaged_data, ['main_ingredient', 'nutritional_information', 'packaging_type'])
packaged_model, compliance_encoder = train_packaged_food_model(packaged_data)

# Encode vegetable data
vegetable_label_encoders = encode_columns(vegetable_data, ['weather_conditions', 'freshness_level', 'storage_value', 'pest_control_methods', 'nutrient_levels', 'safety_certification'])
vegetable_model, vegetable_features, vegetable_target = train_vegetable_model(vegetable_data)

# Streamlit Interface
st.title("Food Compliance and Safety Prediction")

# Section 1: Packaged Food Compliance Prediction
st.header("Packaged Food Compliance Prediction")

main_ingredient = st.selectbox("Select Main Ingredient", packaged_label_encoders['main_ingredient'].classes_)
shelf_life = st.number_input("Enter Shelf Life (in days)", min_value=1, step=1)
nutritional_info = st.selectbox("Select Nutritional Information", packaged_label_encoders['nutritional_information'].classes_)
packaging_type = st.selectbox("Select Packaging Type", packaged_label_encoders['packaging_type'].classes_)

if st.button("Predict Food Compliance"):
    main_ingredient_encoded = packaged_label_encoders['main_ingredient'].transform([main_ingredient])[0]
    nutritional_info_encoded = packaged_label_encoders['nutritional_information'].transform([nutritional_info])[0]
    packaging_type_encoded = packaged_label_encoders['packaging_type'].transform([packaging_type])[0]

    features = np.array([[main_ingredient_encoded, shelf_life, nutritional_info_encoded, packaging_type_encoded]])
    prediction = packaged_model.predict(features)
    compliance_status = compliance_encoder.inverse_transform(prediction)
    
    st.write(f"Compliance Prediction: **{compliance_status[0]}**")

# Section 2: Vegetable Safety Prediction
st.header("Vegetable Safety Prediction")

weather_conditions = st.selectbox("Select Weather Conditions", vegetable_label_encoders['weather_conditions'].classes_)
pH_level = st.text_input("Enter pH Level", value="")
freshness_level = st.selectbox("Select Freshness Level", vegetable_label_encoders['freshness_level'].classes_)
storage_value = st.selectbox("Select Storage Value", vegetable_label_encoders['storage_value'].classes_)
pest_control_methods = st.selectbox("Select Pest Control Methods", vegetable_label_encoders['pest_control_methods'].classes_)
nutrient_levels = st.selectbox("Select Nutrient Levels", vegetable_label_encoders['nutrient_levels'].classes_)

if st.button("Predict Vegetable Safety"):
    user_data = {
        'weather_conditions': vegetable_label_encoders['weather_conditions'].transform([weather_conditions])[0],
        'pH_level': float(pH_level) if pH_level else 0.0,
        'freshness_level': vegetable_label_encoders['freshness_level'].transform([freshness_level])[0],
        'storage_value': vegetable_label_encoders['storage_value'].transform([storage_value])[0],
        'pest_control_methods': vegetable_label_encoders['pest_control_methods'].transform([pest_control_methods])[0],
        'nutrient_levels': vegetable_label_encoders['nutrient_levels'].transform([nutrient_levels])[0]
    }

    user_input = pd.DataFrame([user_data])
    prediction = vegetable_model.predict(user_input[vegetable_features])
    predicted_label = vegetable_label_encoders[vegetable_target].inverse_transform(prediction)[0]
    
    safety_status = "Safe" if predicted_label in ['Certified', 'Compliant', 'Safe'] else "Not Safe"
    st.write(f"Safety Prediction: **{safety_status}**")
