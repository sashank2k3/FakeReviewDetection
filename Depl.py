# LIBRARIES
import streamlit as st
import pickle
import pandas as pd
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# LOAD PICKLE FILES
model = pickle.load(open('data and pickle files/new_model.pkl', 'rb'))
vectorizer = pickle.load(open('data and pickle files/new_vectorizer.pkl', 'rb'))

# FOR STREAMLIT
nltk.download('stopwords')

# TEXT PREPROCESSING
sw = set(stopwords.words('english'))
def text_preprocessing(text):
    logging.debug(f"Original text: {text}")
    txt = TextBlob(text)
    result = txt.correct()
    logging.debug(f"Text after correction: {result}")
    removed_special_characters = re.sub("[^a-zA-Z]", " ", str(result))
    tokens = removed_special_characters.lower().split()
    stemmer = PorterStemmer()
    
    cleaned = [token for token in tokens if token not in sw]
    stemmed = [stemmer.stem(token) for token in cleaned]

    preprocessed_text = " ".join(stemmed)
    logging.debug(f"Preprocessed text: {preprocessed_text}")
    return preprocessed_text

# TEXT CLASSIFICATION
def classify_review(text):
    if len(text) < 1:
        return "Invalid"
    
    cleaned_review = text_preprocessing(text)
    process = vectorizer.transform([cleaned_review]).toarray()
    prediction = model.predict(process)
    return "Genuine" if prediction[0] else "Fraudulent"

# FUNCTION TO STYLE RESULTS
def format_prediction(prediction):
    if prediction == "Genuine":
        return f'<span style="color: green; font-weight: bold;">{prediction}</span>'
    elif prediction == "Fraudulent":
        return f'<span style="color: red; font-weight: bold;">{prediction}</span>'
    else:
        return f'<span style="color: gray; font-weight: bold;">{prediction}</span>'

# PAGE FORMATTING AND APPLICATION
def main():
    st.set_page_config(page_title="Fraud Review Detector", page_icon="🔍", layout="wide")
    
    # Custom CSS for styling
    st.markdown("""
        <style>
            .main-title {
                text-align: center;
                font-size: 30px;
                font-weight: bold;
                color: #2E86C1;
            }
            .sub-title {
                text-align: center;
                font-size: 20px;
                font-weight: bold;
                color: #17A589;
            }
            .uploaded-file {
                font-size: 16px;
                font-weight: bold;
                color: #2874A6;
            }
            .dataframe th {
                background-color: #3498DB;
                color: white;
                text-align: center;
            }
            .dataframe td {
                text-align: center;
            }
            .result-box {
                padding: 10px;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
                font-size: 16px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="main-title">🛒 Fraud Detection in Online Reviews 🔍</p>', unsafe_allow_html=True)
    
    st.markdown('<p class="sub-title">Upload a CSV File with Reviews and Order ID</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], key="file")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)  # Read the CSV
        if 'reviews' not in df.columns or 'orderid' not in df.columns:
            st.error("CSV file must have 'reviews' and 'orderid' columns")
        else:
            df['Verification Status'] = df['orderid'].apply(lambda x: "Verified" if pd.notna(x) and str(x).strip() else "Not a Verified User")
            df['Prediction'] = df.apply(lambda row: classify_review(row['reviews']) if row['Verification Status'] == "Verified" else "Not a Verified User", axis=1)

            # Apply colors to predictions
            df['Prediction'] = df['Prediction'].apply(format_prediction)

            st.write("### Classification Results")
            
            # Display results in a table with alternating row colors
            st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)

# RUN MAIN        
main()
