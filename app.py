
import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("news_category_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

st.set_page_config(page_title="AI News Categorizer", layout="centered")
st.title("🧠 AI-Powered News Categorizer")
st.write("Enter a news article below and the AI will predict its category.")

# Input text
news_input = st.text_area("📰 Paste News Article Here", height=250)

if st.button("🔍 Predict Category"):
    if news_input.strip() == "":
        st.warning("Please enter a news article.")
    else:
        # Transform and predict
        input_vector = vectorizer.transform([news_input])
        prediction = model.predict(input_vector)[0]
        st.success(f"🗂 Predicted Category: **{prediction}**")
