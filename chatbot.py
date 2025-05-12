!pip install streamlit
pip install --force-reinstall transformers
!pip install langchain sentence-transformers faiss-cpu transformers
!pip install -U langchain-community
!pip install python-docx
!pip install tf-keras


# chatbot.py
import json
import streamlit as st
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from transformers import pipeline

# Load JSON files
with open(r"C:\Users\DELL\Downloads\mdcat\json\combined_mdcat_qa.json", "r", encoding="utf-8", errors="ignore") as f:
    past_data = json.load(f)

with open(r"C:\Users\DELL\Downloads\mdcat\json\MDCAT_FAQs.json", "r", encoding="utf-8", errors="ignore") as f:
    faq_data = json.load(f)

# Create Documents
past_docs = [Document(page_content=d["question"] + "\n" + d["answer"]) for d in past_data]
faq_docs = [Document(page_content=d["question"] + "\n" + d["answer"]) for d in faq_data]

# Create Embeddings
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

past_index = FAISS.from_documents(past_docs, embedding)
faq_index = FAISS.from_documents(faq_docs, embedding)

# Load Question Answering pipeline 
#qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")


# Force Hugging Face to use PyTorch backend
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", framework="pt")

# Load QA pipeline (this avoids TensorFlow)
#qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")  # This will now work using PyTorch


def get_answer(query, index):
    docs = index.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    result = qa_pipeline(question=query, context=context)
    return result["answer"], context

