import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth, db as firebase_db
import json
import PyPDF2
from io import BytesIO
from streamlit_pdf_viewer import pdf_viewer


# Initialize Firebase Admin SDK if not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate("/Users/sidharth/unitrial/clinicaltrialproj-firebase-adminsdk-649ug-d2d9e1381c.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://clinicaltrialproj-default-rtdb.firebaseio.com'
    })

# Function to process PDF file
def process_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def display_pdf_viewer(pdf_bytes):
    pdf_viewer(pdf_bytes, height=800)

def login(email, password):
    try:
        # Authenticate user
        user = auth.get_user_by_email(email)
        # No explicit password verification method in Firebase Admin SDK for Python
        return user
    except auth.UserNotFoundError:
        st.error("User not found. Please check your email.")
    except firebase_admin.exceptions.FirebaseError as e:
        st.error(f"Firebase Error: {e}")
    except Exception as e:
        st.error(f"Error: {e}")
    
    return None

def main():
    st.set_page_config(page_title="Clinical Trial Recruitment System", page_icon="🧬", layout="wide")
    st.title("🧬 Clinical Trial Recruitment System")

    st.markdown("""
    Welcome to the Clinical Trial Recruitment System. This app helps you find relevant clinical trials based on your Electronic Health Records (EHR). 
    Simply upload your EHR file (PDF), and we'll do the rest!
    """)

    # User Authentication
    user = None
    user_input = st.text_input("Enter your email:")
    password_input = st.text_input("Enter your password:", type="password")
    if st.button("Login"):
        user = login(user_input, password_input)
        if user:
            st.success(f"Successfully logged in as {user.email}")

    # Upload EHR Data
    st.header("1. Upload Your EHR Data")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf"])

    if uploaded_file is not None and user is not None:
        st.success("File uploaded successfully!")

        # Display PDF viewer
        pdf_bytes = uploaded_file.read()
        display_pdf_viewer(pdf_bytes)
        
        # Process PDF file
        with st.spinner('Processing your PDF file...'):
            text = process_pdf(BytesIO(pdf_bytes))

            # Example: Upload extracted text to Firebase Realtime Database
            ref = firebase_db.reference('/ehr_data')
            ref.update({'pdf_text': text, 'uploaded_by': user.email})
        
        st.header("2. AI Model Prompt")
        prompt = st.text_input("Enter a prompt for the AI model:")
        
        if st.button("Pass to AI Model"):
            if prompt:
                # Placeholder for AI model integration
                st.info(f"Processing with AI model using prompt: '{prompt}'. Replace with actual AI model processing.")
            else:
                st.warning("Please enter a prompt.")

        # Balloons and Success Message
        st.balloons()

if __name__ == "__main__":
    main()

