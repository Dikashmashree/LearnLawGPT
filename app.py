import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time

load_dotenv()

from config import default_api_key
google_api_key = os.getenv("")
if google_api_key is None:
    google_api_key = default_api_key

groq_api_key = os.getenv("")
if groq_api_key is None:
    groq_api_key = default_api_key

st.title("LearnLaw !")

llm = ChatGroq(groq_api_key='', model_name="Llama3-8b-8192")

# Using "context" instead of "documents"
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context, which are laws.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions: {input}
"""
)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=os.getenv("AIzaSyB3pwRnP7wESuoWYiwnwpO8DXBjIt9Kwew")
        )
        st.session_state.loader = PyPDFDirectoryLoader("./dataset")  ## Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  ## Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )  ## Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:20]
        )  # splitting
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings
        )  # vector embeddings

def display_conversation(session_state):
    for i in range(len(session_state["generated"])):
        st.write(f"**You:** {session_state['past'][i]}")
        st.write(f"**Bot:** {session_state['generated'][i]}")

prompt1 = st.text_input("Enter Your Question to know more about Indian Law")

if st.button("Load it"):
    vector_embedding()
    st.write("VectorStore DB is Ready!")

if "past" not in st.session_state:
    st.session_state["past"] = []
if "generated" not in st.session_state:
    st.session_state["generated"] = []

if prompt1:
    # Updated document chain to use "context" as the document_variable_name
    document_chain = create_stuff_documents_chain(llm, prompt, document_variable_name="context")
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    print("Response time:", time.process_time() - start)
    st.session_state["past"].append(prompt1)
    st.session_state["generated"].append(response['answer'])
    st.write(response['answer'])

    # Display conversation history
    if st.session_state["generated"]:
        display_conversation(st.session_state)

    # With a Streamlit expander
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
