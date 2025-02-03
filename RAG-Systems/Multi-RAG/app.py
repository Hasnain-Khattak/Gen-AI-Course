from langchain_community.document_loaders import  PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st

st.set_page_config(page_title="Different Types of RAG", layout="wide")

st.markdown("""
    <h1 style='text-align: center; font-size: 3em; font-weight: bold;'>
        <span style='color: #7ED4E6;'>Chat With Your</span> 
        <span style='color: #FF6666;'>Documents</span>
    </h1>
""", unsafe_allow_html=True)

if "api_key_gemini" not in st.session_state:
    st.session_state.api_key_gemini = ""

if "model_name_gemini" not in st.session_state:
    st.session_state.model_name_gemini = "gemini-1.5-flash"  

if "cohere_api" not in st.session_state:
    st.session_state.cohere_api = ""

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

with st.sidebar:
    st.header('Google Gemini')
    st.write('### You can get the Google Gemini API key From [Here](https://aistudio.google.com/app/apikey)')
    model_name = st.selectbox("Choose LLM:", ["gemini-1.5-flash", "gemini-1.5-pro"], 
                              index=["gemini-1.5-flash", "gemini-1.5-pro"].index(st.session_state.model_name_gemini))
    api_key = st.text_input('Enter Your API Key:', value=st.session_state.api_key_gemini, type="password")

    if api_key and api_key != st.session_state.api_key_gemini:
        st.session_state.api_key_gemini = api_key
    
    if model_name != st.session_state.model_name_gemini:
        st.session_state.model_name_gemini = model_name



if 'vectorstore' not in st.session_state:
    # File uploader for both PDF and DOCX
    files = st.sidebar.file_uploader('Upload Your Documents', type=['pdf', 'docx'], accept_multiple_files=True)

    if files:
        docs = []

        for file in files:
            # Determine file type
            ext = file.name.split('.')[-1].lower()
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_file:
                tmp_file.write(file.read())
                tmp_file_path = tmp_file.name
            
            try:
                # Load file based on type
                if ext == "pdf":
                    loader = PyPDFLoader(tmp_file_path)
                elif ext == "docx":
                    loader = Docx2txtLoader(tmp_file_path)
                else:
                    st.error(f"Unsupported file type: {ext}")
                    continue

                docs.extend(loader.load())  

            finally:
                os.remove(tmp_file_path)

        splitter = RecursiveCharacterTextSplitter()
        chunks = splitter.split_documents(docs)

        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        st.session_state["vectorstore"] = FAISS.from_documents(chunks, embedding)
        st.success("Documents processed and vector store created!")
    else:
        st.warning("Please upload PDF or DOCX files to proceed.")

def reset_chat():
    if len(st.session_state.messages) > 1:
        del st.session_state.messages
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]


col1, col2 = st.columns([1, 5])
for message in st.session_state.messages:
        avatar = "images/male.png" if message["role"] == "user" else "images/scientist.png"
        st.chat_message(message["role"], avatar=avatar).write(message["content"])


with col1:
    st.button("Reset", use_container_width=True, on_click=reset_chat)

if 'vectorstore' in st.session_state:
    if question := st.chat_input("Type your question"):
        st.session_state.messages.append({"role": "user", "content": question})

    
    Page1 = st.Page("pages/hyde.py", title="HYDE RAG", icon=":material/science:")
    Page2 = st.Page("pages/multiquery.py", title="MultiQuery RAG", icon=":material/quiz:")
    Page3 = st.Page("pages/reranker.py", title="Reranker RAG", icon=":material/trending_up:")
        
        
    current_page = st.navigation({"Rag Techniques:":[Page1, Page2, Page3]})
    current_page.run()