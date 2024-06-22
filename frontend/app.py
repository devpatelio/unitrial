import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import json

# Streamlit app layout
st.set_page_config(page_title="Clinical Trial Recruitment System", page_icon="🧬", layout="wide")

st.title("🧬 Clinical Trial Recruitment System")

st.markdown("""
Welcome to the Clinical Trial Recruitment System. This app helps you find relevant clinical trials based on your Electronic Health Records (EHR). 
Simply upload your EHR file, and we'll do the rest!
""")

st.header("1. Upload Your EHR Data")
uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt", "json"])

if uploaded_file is not None:
    st.success("File uploaded successfully!")
    
    # Mock data processing
    with st.spinner('Processing your data...'):
        ehr_data = uploaded_file.read().decode("utf-8")  # Read file content
        formatted_data = preprocess_ehr_data(ehr_data)
        embeddings = create_embeddings(formatted_data)
    
        clinical_trials = fetch_clinical_trials()
        filtered_trials = filter_trials(embeddings, clinical_trials)
        relevant_trials = calculate_similarity_scores(embeddings, filtered_trials)
    
    st.header("2. Relevant Clinical Trials")
    for trial in relevant_trials:
        st.write(f"🔍 {trial}")

    st.balloons()
else:
    st.info("Please upload your EHR file to proceed.")

    