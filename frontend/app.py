import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth, db as firebase_db
import json
import PyPDF2
from io import BytesIO
from streamlit_pdf_viewer import pdf_viewer

import os
import chromadb
from langchain import hub
import ollama
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.embeddings import HuggingFaceBgeEmbeddings

import torch

# Function to process PDF file
def process_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

bge_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en-v1.5", encode_kwargs={'normalize_embeddings': True})

clinical_trials_db = Chroma(persist_directory="../chroma_db", embedding_function=bge_model)

def dot_product(vec1, vec2):
    return sum(a * b for a, b in zip(vec1, vec2))

def magnitude(vec):
    return (sum(v**2 for v in vec)) ** 0.5

def cosine_similarity(vec1, vec2):
    dot_prod = dot_product(vec1, vec2)
    mag_vec1 = magnitude(vec1)
    mag_vec2 = magnitude(vec2)

    if mag_vec1 == 0 or mag_vec2 == 0:
        return 0  # Handle division by zero

    return dot_prod / (mag_vec1 * mag_vec2)


#Takes int k, query q, and database of chunks db and returns top k chunks based on similarity score
def topK(k, q, db):
    similarities = [(cosine_similarity(q, x), x) for x in db]
    similarities.sort(key = lambda x: x[0], reverse=True)

    return similarities[:k][1]

def getTopK(k, q, db, ehr_data):
    #llm = Ollama(model="mistral", temperature=0.7) # setup llm
    model = ChatMistralAI(api_key="qXTo15UgUxahwAGNHMSI7dVJ2NKJaahf", temperature=0)


    rag_prompt_llama = hub.pull("rlm/rag-prompt-llama") # setup prompt
    template = PromptTemplate.from_template(
            """
            [INST]<<SYS>>  You are an assistant for helping patients find clinical trials based on their profiles. Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know. Never, ever make stuff up since this is a matter of life and death for patients. Return AT MOST 3 trials. Use three sentences maximum and keep the answer concise. ENSURE YOUR ANSWER
            ALWAYS IS THE EXACT TITLE OF THE TRIAL. Nothing should be changed. Specifically, look for Title keyword followed by the title of the study itself. Also ensure
            you are providing the correct trial ID identifier. Context: {ehr_data} <</SYS>>
            Question: {question}
            Answer: [/INST]
            """)
    #f"You are a smart agent that seeks to match patients with clinical trials. A question would be asked to you and relevant information would be provided.\
    #Your task is to answer the question using the information provided. Be extremely concise and use five sentences maximum. Question: {question} and Context: {context}"


    retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                'k': 1,
                'score_threshold': 0.5
            },
    )

    qa_chain = ({"context" : retriever,
            "question" : RunnablePassthrough()}
                    | template | model  | StrOutputParser()
    )



    # Use db.as_retriever() for retrieval
    #qa_chain = RetrievalQA.from_chain_type(model, retriever=db.as_retriever(),
    #            chain_type_kwargs={"prompt": rag_prompt_llama})

    return qa_chain.invoke(q)
# Initialize Firebase Admin SDK if not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate("../clinicaltrialproj-firebase-adminsdk-649ug-d2d9e1381c.json")
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

    if uploaded_file is not None:
        st.success("File uploaded successfully!")

        # Display PDF viewer
        pdf_bytes = uploaded_file.read()
        display_pdf_viewer(pdf_bytes)
        
        pdf_string = process_pdf(uploaded_file)
        # Process PDF file
        with st.spinner('Processing your PDF file...'):
            text = process_pdf(BytesIO(pdf_bytes))

            # Example: Upload extracted text to Firebase Realtime Database
            ref = firebase_db.reference('/ehr_data')
            ref.update({'pdf_text': text})
        
        st.header("2. AI Model Prompt")
        
        if st.button("Pass to AI Model"):
            prompt = st.text_input("Enter a prompt for the AI model:")
            if prompt:
                # Placeholder for AI model integration
                st.info(f"Processing with AI model using prompt: '{prompt}'. Replace with actual AI model processing.")
                result = getTopK(3, str(prompt), clinical_trials_db, pdf_string)
                st.write(result)
            else:
                st.warning("Please enter a prompt.")

        
        # Balloons and Success Message
        st.balloons()
        
    


if __name__ == "__main__":
    main()


