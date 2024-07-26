

import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import streamlit as st

# Display title 
st.title("Email Spam & Ham Detection")

# Take email input
email = st.text_input("Enter email for spam detection:")

# Add helpful message
st.markdown("""
**Tips for testing:**
- For likely Ham detection, include words like: lt, will, gt, u, ok
- For likely Spam detection, include words like: free, text, call
""")

# Load the saved vectorizer
vectorizer = pickle.load(open('./vectorizer.pkl', 'rb'))

# Load the saved model
model = pickle.load(open('./email_spam_ham.pkl', 'rb'))

if email:
    # Transform the input email
    Numerical_data = vectorizer.transform([email]).toarray()

    # Prediction
    result = model.predict(Numerical_data)[0]

    # Print the prediction
    st.write("Provided email is:", "Spam" if result == 1 else "Ham")

# Optional: Display some example messages
st.markdown("---")
st.subheader("Example messages to try:")
st.markdown("1. Ham example: 'Will u be home by 6? Ok if not, I'll wait.'")
st.markdown("2. Spam example: 'Free text alert! Call now to claim your prize!'")