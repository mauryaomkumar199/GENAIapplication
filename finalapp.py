import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

# Load the NVIDIA API KEY
api_key = os.getenv('NVIDIA_API_KEY')
if not api_key:
    st.error("NVIDIA_API_KEY is missing. Please check your .env file.")
else:
    st.write("NVIDIA API Key loaded successfully!")
os.environ['NVIDIA_API_KEY'] = api_key

llm = ChatNVIDIA(model="meta/llama-3.2-3b-instruct")

def vector_embedding():
    if "vectors" not in st.session_state:
        try:
            if "embeddings" not in st.session_state:
                st.session_state.embeddings = NVIDIAEmbeddings()
            st.session_state.loader = PyPDFDirectoryLoader("./news")
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=70, chunk_overlap=50)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents, st.session_state.embeddings
            )
            st.write("Vector embeddings initialized successfully!")
        except Exception as e:
            st.error(f"Error during vector embedding initialization: {e}")

st.title("Nvidia NIM Demo")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    Question: {input}
    """
)

prompt1 = st.text_input("Enter your Question From Documents")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("FAISS Vector Store DB is ready using NvidiaEmbedding")

import time

if prompt1:
    if "vectors" not in st.session_state:
        st.warning("Please initialize the vector database first by clicking the button above.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        start = time.process_time()
        try:
            response = retrieval_chain.invoke({"input": prompt1})
            st.write("Response time:", time.process_time() - start)
            st.write(response["answer"])
            
            with st.expander("Documents Similarity Search"):
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("--------------------------")
        except Exception as e:
            st.error(f"Error during retrieval: {e}")
