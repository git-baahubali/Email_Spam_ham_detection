import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer


# def get_vocabulary(Series):
#     vectorizer = CountVectorizer()
#     vectorizer.fit(Series)
#     vocabulary = vectorizer.get_feature_names_out()
#     return vocabulary

import pickle
import streamlit as st

# Display tile 
st.title = "Email Spam & Ham Detection"

# Take email input
default_value_for_email_testing ="Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
email = st.text_input("enter email for spam detection:", value =default_value_for_email_testing)
print("type of email")
print(type(email), )
print(f"email:{email}")

# !------>This method is deprecated
# # Load dataset
# df = pd.read_csv(r'./spam.csv')

# ## Covert the email in text form to numerical data 
# vectorizer = CountVectorizer(stop_words='english',vocabulary=get_vocabulary(df['Message']) )

# Numerical_data = vectorizer.transform([email]).toarray() 
# !                                <----


# Load vectorizer and model
vectorizer = pickle.load( open(r'vectorizer.pkl','rb') )
model = pickle.load( open(r'./email_spam_ham.pkl','rb') )

if email:
    Numerical_data = vectorizer.transform([email]).toarray() 
    result = model.predict(Numerical_data)[0]
    
    #Print the prediction
    st.write("Provided email is :",result)
