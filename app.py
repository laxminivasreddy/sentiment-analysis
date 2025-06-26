import streamlit as st 
import joblib 
import re 
import nltk 
from nltk.corpus import stopwords


nltk.download("stopwords")
stop_words = set(stopwords.words('english'))

#Load the model and Vectorizer
model = joblib.load("sentiment_analysis_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

#clean text
def clean_text(text):
    text = re.sub(r"[^a-zA-Z]"," ",text).lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)


# Create UI
st.set_page_config(page_title = "Sentiment Analysis for Movies", layout = "centered" )
st.title("Sentiment Analysis app for movies")
st.markdown ("Enter a movie")

user_input = st.text_area("Enter the review")

if st.button ("Predict Sentiment"):
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    sentiment = "positive" if prediction == 1 else "negative"
    st.success(f"The sentiment of the review is: {sentiment}")
