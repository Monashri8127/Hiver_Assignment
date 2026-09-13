import streamlit as st
import joblib

# Load your trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Customer Support Text Classifier")

# Input box
user_input = st.text_input("Enter a customer query:")

if user_input:
    X = vectorizer.transform([user_input])
    prediction = model.predict(X)[0]
    st.write(f"Predicted Category: **{prediction}**")
