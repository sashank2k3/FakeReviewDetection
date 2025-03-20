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

# FUNCTION TO CHECK VERIFICATION
def check_verification(orderid, retailer, productname):
    valid_retailers = ["amazon", "flipkart", "myntra", "meesho", "ajio"]
    
    if (
        pd.notna(orderid) and str(orderid).strip() and
        pd.notna(retailer) and str(retailer).strip().lower() in valid_retailers and
        pd.notna(productname) and str(productname).strip()
    ):
        return "✅ Verified User"
    else:
        return "❌ Not a Verified User"

# FUNCTION TO STYLE RESULTS
def format_prediction(prediction):
    if "Genuine" in prediction:
        return f'<span class="genuine">{prediction}</span>'
    elif "Fraudulent" in prediction:
        return f'<span class="fraudulent">{prediction}</span>'
    else:
        return f'<span class="unverified">⚪ Not a Verified User</span>'

# PAGE FORMATTING AND APPLICATION
def main():
    st.set_page_config(page_title="Fraud Review Detector", page_icon="🔍", layout="wide")
    
    # Improved CSS for Better Readability
    st.markdown("""
        <style>
            body { font-family: 'Arial', sans-serif; }
            .main-title {
                text-align: center;
                font-size: 36px;
                font-weight: bold;
                color: white;
                padding: 15px;
                background: linear-gradient(to right, #1B1B1B, #2C3E50);
                border-radius: 10px;
            }
            .sub-title {
                text-align: center;
                font-size: 22px;
                color: #1ABC9C;
                margin-top: -15px;
            }
            .card {
                background: #ffffff;
                padding: 15px;
                border-radius: 10px;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                margin: 10px 0;
                color: black;  /* Ensure text is readable */
            }
            .review-text {
                font-size: 18px;
                font-weight: bold;
                color: #2C3E50;
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
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="main-title">🛒 Fraud Detection in Online Reviews 🔍</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Upload a CSV File with Reviews, Order ID, Retailer & Product Name</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], key="file")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)  # Read the CSV
        if not all(col in df.columns for col in ["reviews", "orderid", "retailer", "productname"]):
            st.error("CSV file must have 'reviews', 'orderid', 'retailer', and 'productname' columns")
        else:
            df['Verification Status'] = df.apply(lambda row: check_verification(row['orderid'], row['retailer'], row['productname']), axis=1)
            df['Prediction'] = df.apply(lambda row: classify_review(row['reviews']) if row['Verification Status'] == "✅ Verified User" else "⚪ Not a Verified User", axis=1)

            # Apply styles to predictions
            df['Styled Prediction'] = df['Prediction'].apply(format_prediction)

            st.write("### Classification Results")
            
            # Display results using custom-styled cards
            for i, row in df.iterrows():
                st.markdown(f"""
                    <div class="card">
                        <p class="review-text">📌 <b>Review:</b> {row['reviews']}</p>
                        <p><b>🆔 Order ID:</b> <span style="color:#2874A6;">{row['orderid'] if row['orderid'] else 'N/A'}</span></p>
                        <p><b>🏬 Retailer:</b> <span style="color:#F39C12;">{row['retailer'] if row['retailer'] else 'N/A'}</span></p>
                        <p><b>📦 Product Name:</b> <span style="color:#16A085;">{row['productname'] if row['productname'] else 'N/A'}</span></p>
                        <p><b>🔍 Verification Status:</b> <span style="color:#1ABC9C;">{row['Verification Status']}</span></p>
                        <p><b>🔎 Prediction:</b> {row['Styled Prediction']}</p>
                    </div>
                """, unsafe_allow_html=True)

# RUN MAIN        
main()
