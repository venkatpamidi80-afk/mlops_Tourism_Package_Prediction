import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_predection_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Prediction App")
st.write("""
This application predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them.
""")

# User Inputs
 
age = st.number_input("Age", 18, 100, 35)
 
type_of_contact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)
 
city_tier = st.selectbox("City Tier", [1, 2, 3])
 
occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Free Lancer", "Large Business"]
)
 
gender = st.selectbox("Gender", ["Male", "Female"])
 
duration_of_pitch = st.number_input("Duration of Pitch", 1, 60, 15)
 
number_of_persons_visiting = st.number_input("Persons Visiting", 1, 10, 2)
 
number_of_followups = st.number_input("Follow-ups", 0, 10, 3)
 
product_pitched = st.selectbox(
    "Product",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)
 
preferred_property_star = st.selectbox(
    "Property Rating",
    [3.0, 4.0, 5.0]
)
 
marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced", "Unmarried"]
)
 
number_of_trips = st.number_input("Trips Per Year", 0, 20, 3)
 
passport = st.selectbox("Passport", ["Yes", "No"])
 
pitch_score = st.slider("Pitch Satisfaction", 1, 5, 3)
 
own_car = st.selectbox("Own Car", ["Yes", "No"])
 
children = st.number_input("Children Visiting", 0, 5, 0)
 
designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)
 
monthly_income = st.number_input(
    "Monthly Income",
    min_value=0,
    value=25000
)
 
# Encoding
 
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": {"Company Invited":0,"Self Enquiry":1}[type_of_contact],
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": {"Free Lancer":0,"Large Business":1,"Salaried":2,"Small Business":3}[occupation],
    "Gender": {"Female":0,"Male":1}[gender],
    "NumberOfPersonVisiting": number_of_persons_visiting,
    "NumberOfFollowups": number_of_followups,
    "ProductPitched": {"Basic":0,"Deluxe":1,"King":2,"Standard":3,"Super Deluxe":4}[product_pitched],
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": {"Divorced":0,"Married":1,"Single":2,"Unmarried":3}[marital_status],
    "NumberOfTrips": number_of_trips,
    "Passport": 1 if passport=="Yes" else 0,
    "PitchSatisfactionScore": pitch_score,
    "OwnCar": 1 if own_car=="Yes" else 0,
    "NumberOfChildrenVisiting": children,
    "Designation": {"AVP":0,"Executive":1,"Manager":2,"Senior Manager":3,"VP":4}[designation],
    "MonthlyIncome": monthly_income
}])

if st.button("Tourism Predict"):
 
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
 
    if prediction == 1:
        st.success("Customer is likely to purchase the tourism package.")
    else:
        st.warning("Customer is unlikely to purchase the tourism package.")
 
    st.write(f"Purchase Probability: **{probability:.2%}**")
