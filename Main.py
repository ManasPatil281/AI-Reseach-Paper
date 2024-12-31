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


about_page = st.Page(
    "Paraphraser.py",
    title="Paraphraser",
    icon=":material/account_circle:",
    default=True,
)
gram= st.Page(
    "Grammar checker.py",
    title="Grammar Checker",
    icon=":material/account_circle:",
)
plag=st.Page(
    "Plagiarism.py",
    title="Plagiarism Checker",
    icon=":material/account_circle:",
)


# --- NAVIGATION SETUP [WITHOUT SECTIONS] ---
# pg = st.navigation(pages=[about_page, project_1_page, project_2_page])

# --- NAVIGATION SETUP [WITH SECTIONS]---
pg = st.navigation(
    {
        "Info": [about_page],
        "Grammar Checker": [gram],
        "Plagiarism Checker": [plag],

    }
)


# --- SHARED ON ALL PAGES ---


st.sidebar.markdown("Made with ❤️ by ")
# --- RUN NAVIGATION ---
pg.run()