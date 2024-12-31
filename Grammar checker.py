import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd
import json
from io import BytesIO
import tempfile
import webbrowser
import random

load_dotenv()



HF_TOKEN = os.getenv('HF_TOKEN')
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# API keys
api_key = os.getenv('API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API')
os.environ["GOOGLE_API_KEY"] = 'AIzaSyDzJdYXogH--ueNwQF3z9o85Ln-iuOmG_s'

#if "GOOGLE_API_KEY" not in os.environ:
    #os.environ["GOOGLE_API_KEY"] ="AIzaSyDzJdYXogH--ueNwQF3z9o85Ln-iuOmG_s"

llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")
llm_2 = ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it")
llm3 = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash"
)


option = st.selectbox(
    "Select Language of your choice",
    ("English",
    "Mandarin Chinese",
    "Spanish",
    "Hindi",
    "Arabic",
    "Bengali",
    "Portuguese",
    "Russian",
    "Japanese",
    "Punjabi",
    "German",
    "Korean",
    "French",
    "Turkish",
    "Vietnamese",
    "Italian",
    "Persian",
    "Swahili",
    "Tamil",
    "Urdu")
)

type=st.selectbox(
    "Select mode of your choice",
    (  "Reserach Paper",       # Default mode, balances fluency and rewriting
    "Email",        # Focuses on improving grammar and fluency
    "Essay",         # Makes the text more professional and polished
    "Buisness Memo",         # Simplifies the text, suitable for better readability
    "Marketing Copy",       # Provides more diverse and imaginative rewrites        # Reduces the length of the text while keeping the key message),
))


text = st.text_input("Enter the text you want to paraphrase") or None
file = st.file_uploader("Upload a file", type="pdf") or None


def gram_checker(files,text, mode, language):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(files.getvalue())
            temp_pdf_path = temp_pdf.name

        # Load the PDF using PyPDFLoader
        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()

    finally:
        # Remove the temporary file after processing
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

    # Text splitting for embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(docs)

    # Create FAISS vectorstore for retrieval
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()

    system_prompt = (
        f"Paraphrase the following text or file {text} using the '{mode}' mode in '{language}' language."
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{context}\n{input}"),
    ])


    try:
        # Assume LLM object is preloaded
        question_answer_chain = create_stuff_documents_chain(llm3, qa_prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        response = rag_chain.invoke({"input":'paraphrase the text'})
        paraphrased_text = response["answer"]
        return paraphrased_text
    except Exception as e:
        raise RuntimeError(f"Error paraphrasing text: {e}")

if st.button("Check Grammar"):
    output = gram_checker(file, text,type, option)
    st.write_stream(output)

