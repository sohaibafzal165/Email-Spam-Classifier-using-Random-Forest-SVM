import streamlit as st
import joblib
import numpy as np

# Load vectorizer once
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Load both models
model_svm = joblib.load("spam_classifier_svm_model.pkl")
model_rf = joblib.load("spam_classifier_rf_model.pkl")

# Streamlit page config
st.set_page_config(page_title="Email Spam Classifier", page_icon="📧", layout="centered")

# -------- UI Design --------
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📧 Email Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #ccc'>", unsafe_allow_html=True)

# Model selection
model_choice = st.selectbox("🎯 Choose Classification Model", ["SVM", "Random Forest"])

# Message input
email_input = st.text_area("✉️ Enter your email message below:", height=150, placeholder="Write or paste email content here...")

# Predict button
if st.button("🔍 Predict"):
    if not email_input.strip():
        st.warning("⚠️ Please enter a message to classify.")
    else:
        # Vectorize input
        transformed_input = vectorizer.transform([email_input])

        # Predict based on model choice
        if model_choice == "SVM":
            prediction = model_svm.predict(transformed_input)[0]
        else:
            prediction = model_rf.predict(transformed_input)[0]

        # Display result
        if prediction == 1 or prediction == 'spam':
            st.error("🚨 This email is classified as: **SPAM**")
        else:
            st.success("✅ This email is classified as: **NOT SPAM**")

# Footer
st.markdown("<hr style='border:1px solid #ccc'>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: gray;'>Developed by Sohaib | Powered by Streamlit</div>", unsafe_allow_html=True)

#python -m streamlit run app.py