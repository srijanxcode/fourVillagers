import pandas as pd
import mysql.connector
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Database connection
def get_data_from_db():
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='mysqlyug',
        database='h4hack'
    )
    query = "SELECT * FROM vegetable_data"
    df = pd.read_sql(query, connection)
    connection.close()
    return df

# Load the data
df = get_data_from_db()

# Display the DataFrame to help troubleshoot
st.write("DataFrame contents:", df)

# Select relevant features
features = ['weather_conditions', 'pH_level', 'freshness_level', 'storage_value', 
            'pest_control_methods', 'nutrient_levels']

# Target variable
target = 'safety_certification'

# Preprocess categorical data
label_encoders = {}
for col in features + [target]:
    if df[col].dtype == 'object':  # Only encode categorical features
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Split data into features (X) and target (y)
X = df[features]
y = df[target]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Streamlit app
st.title("Food Safety Prediction")

# Create dropdowns for user input based on unique values from the database
weather_conditions = label_encoders['weather_conditions'].inverse_transform(np.arange(len(label_encoders['weather_conditions'].classes_))).tolist()
freshness_levels = label_encoders['freshness_level'].inverse_transform(np.arange(len(label_encoders['freshness_level'].classes_))).tolist()
storage_values = label_encoders['storage_value'].inverse_transform(np.arange(len(label_encoders['storage_value'].classes_))).tolist()
pest_control_methods = label_encoders['pest_control_methods'].inverse_transform(np.arange(len(label_encoders['pest_control_methods'].classes_))).tolist()
nutrient_levels = label_encoders['nutrient_levels'].inverse_transform(np.arange(len(label_encoders['nutrient_levels'].classes_))).tolist()

weather = st.selectbox("Select Weather Conditions", weather_conditions)
pH_level = st.text_input("Enter the pH Level", value="")  # Allow keyboard input for pH level
freshness = st.selectbox("Select Freshness Level", freshness_levels)
storage = st.selectbox("Select Storage Value", storage_values)
pest_control = st.selectbox("Select Pest Control Methods", pest_control_methods)
nutrient_level = st.selectbox("Select Nutrient Levels", nutrient_levels)

# Prepare the data for prediction
user_data = {
    'weather_conditions': weather,
    'pH_level': float(pH_level) if pH_level else 0.0,  # Convert to float, default to 0.0 if empty
    'freshness_level': freshness,
    'storage_value': storage,
    'pest_control_methods': pest_control,
    'nutrient_levels': nutrient_level
}

# Encode categorical input data
for feature in features:
    if feature in label_encoders:
        if feature == 'pH_level':
            # Keep pH level as is, do not encode
            user_data[feature] = user_data[feature]
        else:
            user_data[feature] = label_encoders[feature].transform([user_data[feature]])[0]

# Make a prediction
user_input = pd.DataFrame([user_data])
prediction = clf.predict(user_input[features])

# Decode the prediction back to the original label
predicted_label = label_encoders[target].inverse_transform(prediction)[0]

# Output the result
safety_status = "Safe" if predicted_label in ['Certified', 'Compliant', 'Safe'] else "Not Safe"
st.write(f"The produce is predicted to be: **{safety_status}**")
