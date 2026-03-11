import streamlit as st
from src.prediction import Insurance_prediction
st.title("Insurance Premium Prediction")
st.write("Enter the details to predict the annual premium :")

age=st.number_input("Age :")
annual_income=st.number_input("Annual Income  :")
policy_term=st.number_input("Policy Term  :")
sum_assured=st.number_input("Sum Assured  :")

if st.button("Predict"):
    model=Insurance_prediction()
    result=model.prediction(age,annual_income,policy_term,sum_assured)
    st.success(result)

 
 