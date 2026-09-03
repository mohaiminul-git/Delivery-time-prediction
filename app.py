import pandas as pd
import streamlit as st

from src.config.configuration import (
    FEATURE_ENGINEERED_TRAIN_DATA_PATH,
    MODEL_TRAINER_PATH,
    PREPROCESSING_OBJ_PATH,
)
from src.utils import load_model

preprocessor = load_model(PREPROCESSING_OBJ_PATH)
model = load_model(MODEL_TRAINER_PATH)

st.title("Delivery Time Prediction App 🚀")
st.sidebar.header("Enter Delivery Details")

df = pd.read_csv(FEATURE_ENGINEERED_TRAIN_DATA_PATH)

Type_of_order = st.sidebar.selectbox("Type of Order", df["Type_of_order"].unique())
Type_of_vehicle = st.sidebar.selectbox("Type of Vehicle", df["Type_of_vehicle"].unique())
Delivery_city = st.sidebar.selectbox("Delivery City", df["Delivery_city"].unique())

Road_traffic_density = st.sidebar.selectbox("Road Traffic Density", df["Road_traffic_density"].unique())
Weather_conditions = st.sidebar.selectbox("Weather Conditions", df["Weather_conditions"].unique())
Delivery_person_Age = st.sidebar.number_input(
    "Delivery Person Age", df["Delivery_person_Age"].min(), df["Delivery_person_Age"].max(),
    df["Delivery_person_Age"].median(),
)
Delivery_person_Ratings = st.sidebar.slider(
    "Delivery Person Ratings", df["Delivery_person_Ratings"].min(), df["Delivery_person_Ratings"].max(),
    df["Delivery_person_Ratings"].median(),
)
Vehicle_condition = st.sidebar.selectbox("Vehicle Condition", df["Vehicle_condition"].unique())
multiple_deliveries = st.sidebar.slider(
    "Multiple Deliveries", df["multiple_deliveries"].min(), df["multiple_deliveries"].max(),
    df["multiple_deliveries"].median(),
)
Time_Orderd_hour = st.sidebar.slider(
    "Time Ordered (Hour)", df["Time_Orderd_hour"].min(), df["Time_Orderd_hour"].max(),
    df["Time_Orderd_hour"].median(),
)
distance = st.sidebar.slider("Distance (km)", df["distance"].min(), df["distance"].max(), df["distance"].median())
Festival = st.sidebar.selectbox("Is it a Festival?", options=["Yes", "No"])
City = st.sidebar.selectbox("City Type", options=["Metropolitian", "Urban", "Semi-Urban"])

input_data = pd.DataFrame({
    "Type_of_order": [Type_of_order],
    "Type_of_vehicle": [Type_of_vehicle],
    "Festival": [Festival],
    "City": [City],
    "Delivery_city": [Delivery_city],
    "Road_traffic_density": [Road_traffic_density],
    "Weather_conditions": [Weather_conditions],
    "Delivery_person_Age": [Delivery_person_Age],
    "Delivery_person_Ratings": [Delivery_person_Ratings],
    "Vehicle_condition": [Vehicle_condition],
    "multiple_deliveries": [multiple_deliveries],
    "Time_Orderd_hour": [Time_Orderd_hour],
    "distance": [distance],
})

if st.button("Predict Delivery Time"):
    try:
        data_scaled = preprocessor.transform(input_data)
        prediction = model.predict(data_scaled)
        st.success(f"Estimated Delivery Time: {round(prediction[0], 2)} minutes ⏳")
    except Exception as e:
        st.error(f"Error: {e}")
