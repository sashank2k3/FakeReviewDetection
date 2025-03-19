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
model = pickle.load(open('new_model.pkl', 'rb'))
vectorizer = pickle.load(open('new_vectorizer.pkl', 'rb'))

# FOR STREAMLIT
nltk.download('stopwords')

# TEXT PREPROCESSING
sw = set(stopwords.words('english'))
def text_preprocessing(text):
    txt = TextBlob(text)
    result = txt.correct()
    removed_special_characters = re.sub("[^a-zA-Z]", " ", str(result))
    tokens = removed_special_characters.lower().split()
    stemmer = PorterStemmer()
    
    cleaned = [token for token in tokens if token not in sw]
    stemmed = [stemmer.stem(token) for token in cleaned]

    return " ".join(stemmed)

# TEXT CLASSIFICATION
def classify_review(text):
    if len(text) < 1:
        return "Invalid"
    
    cleaned_review = text_preprocessing(text)
    process = vectorizer.transform([cleaned_review]).toarray()
    prediction = model.predict(process)
    return "🟢 Genuine" if prediction[0] else "🔴 Fraudulent"

# FUNCTION TO STYLE RESULTS
def format_prediction(prediction):
    if "Genuine" in prediction:
        return f'<p class="genuine">{prediction}</p>'
    elif "Fraudulent" in prediction:
        return f'<p class="fraudulent">{prediction}</p>'
    else:
        return f'<p class="unverified">⚪ Not a Verified User</p>'

# PAGE FORMATTING AND APPLICATION
def main():
    st.set_page_config(page_title="Fraud Review Detector", page_icon="🔍", layout="wide")
    
    # Custom CSS for Styling
    st.markdown("""
        <style>
            body { font-family: 'Arial', sans-serif; }
            .main-title {
                text-align: center;
                font-size: 36px;
                font-weight: bold;
                color: #ffffff;
                padding: 10px;
                background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
                border-radius: 10px;
            }
            .sub-title {
                text-align: center;
                font-size: 22px;
                color: #1ABC9C;
                margin-top: -15px;
            }
            .uploaded-file {
                font-size: 18px;
                font-weight: bold;
                color: #2874A6;
            }
            .dataframe {
                border-radius: 10px;
                overflow: hidden;
                background: white;
                box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.1);
            }
            .dataframe th {
                background-color: #2E86C1;
                color: white;
                text-align: center;
                padding: 12px;
            }
            .dataframe td {
                text-align: center;
                padding: 10px;
            }
            .genuine {
                color: #27AE60;
                font-weight: bold;
                font-size: 16px;
            }
            .fraudulent {
                color: #E74C3C;
                font-weight: bold;
                font-size: 16px;
            }
            .unverified {
                color: #7F8C8D;
                font-weight: bold;
                font-size: 16px;
            }
            .card {
                background: #F7F9F9;
                padding: 15px;
                border-radius: 10px;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                margin: 10px 0;
            }
            .review-text {
                font-size: 18px;
                font-weight: bold;
                color: #2C3E50;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="main-title">🛒 Fraud Detection in Online Reviews 🔍</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Upload a CSV File with Reviews & Order ID</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], key="file")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)  # Read the CSV
        if 'reviews' not in df.columns or 'orderid' not in df.columns:
            st.error("CSV file must have 'reviews' and 'orderid' columns")
        else:
            df['Verification Status'] = df['orderid'].apply(lambda x: "Verified ✅" if pd.notna(x) and str(x).strip() else "Not a Verified User ❌")
            df['Prediction'] = df.apply(lambda row: classify_review(row['reviews']) if row['Verification Status'] == "Verified ✅" else "⚪ Not a Verified User", axis=1)

            # Apply styles to predictions
            df['Styled Prediction'] = df['Prediction'].apply(format_prediction)

            st.write("### Classification Results")
            
            # Display results using custom cards
            for i, row in df.iterrows():
                st.markdown(f"""
                    <div class="card">
                        <p class="review-text">📌 Review: {row['reviews']}</p>
                        <p><b>🆔 Order ID:</b> {row['orderid'] if row['orderid'] else 'N/A'}</p>
                        <p><b>🔍 Verification Status:</b> {row['Verification Status']}</p>
                        {row['Styled Prediction']}
                    </div>
                """, unsafe_allow_html=True)

# RUN MAIN        
main()
