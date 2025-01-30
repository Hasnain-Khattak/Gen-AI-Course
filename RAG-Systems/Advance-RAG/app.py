from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import tempfile
import os
from langchain.schema import Document

st.set_page_config(page_title='Advance RAG', page_icon='🤖')

st.markdown("""
    <h1 style='text-align: center; font-size: 3em; font-weight: bold;'>
        <span style='color: #7ED4E6;'>Chat With Your</span> 
        <span style='color: #FF6666;'>Documents</span>
    </h1>
""", unsafe_allow_html=True)


google_models = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

# groq_models = [
#     "llama-3.1-8b-instant",
#     "llama-3.1-70b-versatile",
#     "llama3-70b-8192",
#     "llama3-8b-8192",
#     "gemma2-9b-it",
#     "mixtral-8x7b-32768"
# ]

# ###---LLM & PARAMETERS---###
# def get_llm_info(available_models):
#     with st.sidebar:
#         tip =tip = "Select Gemini models if you require multi-modal capabilities (text, image, audio and video inputs)"
#         model = st.selectbox("Choose LLM:", available_models, help=tip)

#         model_type = None
#         if model.startswith(("llama", "gemma", "mixtral")): model_type = "groq"
#         elif model.startswith("gemini"): model_type = "google"

#         with st.popover("⚙️Model Parameters", use_container_width=True):
#             temp = st.slider("Temperature:", min_value=0.0,
#                                             max_value=2.0, value=0.5, step=0.5)
            
#             max_tokens = st.slider("Maximum Tokens:", min_value=100,
#                                         max_value=2000, value=400, step=200)
#     return model, model_type, temp, max_tokens



# # Initialize session state variables if not already set
# if "api_key_gemini" not in st.session_state:
#     st.session_state["api_key_gemini"] = ""

# if "api_key_groq" not in st.session_state:
#     st.session_state["api_key_groq"] = ""

# # Callback function to update API key state
# def update_api_key(api_type):
#     if api_type == "gemini":
#         st.session_state["api_key_groq"] = ""  # Clear Groq key when Gemini key is entered
#     elif api_type == "groq":
#         st.session_state["api_key_gemini"] = ""  # Clear Gemini key when Groq key is entered

# # Sidebar setup
# with st.sidebar:
#     st.header("API Configuration")

#     # Create two equal-sized columns inside the sidebar
#     col1, col2 = st.columns(2)

#     # Google Gemini Section (Visible only if Groq API is not entered)
#     if not st.session_state["api_key_groq"]:
#         with col1:
#             st.markdown(
#                 "<div style='padding: 10px; text-align: center; background-color: rgba(255,255,255,0.1); border-radius: 10px;'>"
#                 "<a href='https://aistudio.google.com/app/apikey' target='_blank' style='text-decoration: none; font-size: 16px; color: white;'>Google Gemini</a>"
#                 "</div>",
#                 unsafe_allow_html=True
#             )

#     # Groq Section (Visible only if Gemini API is not entered)
#     if not st.session_state["api_key_gemini"]:
#         with col2:
#             st.markdown(
#                 "<div style='padding: 10px; text-align: center; background-color: rgba(255,255,255,0.1); border-radius: 10px;'>"
#                 "<a href='https://console.groq.com/keys' target='_blank' style='text-decoration: none; font-size: 16px; color: white;'>Groq</a>"
#                 "</div>",
#                 unsafe_allow_html=True
#             )

#     st.divider()

#     # API Key Input & Model Selection (Google Gemini)
#     if not st.session_state["api_key_groq"]:
#         st.text_input("Enter Google Gemini API Key:", type="password", key="api_key_gemini", on_change=update_api_key, args=("gemini",))
#         if st.session_state["api_key_gemini"]:
#             get_llm_info(google_models)

#     # API Key Input & Model Selection (Groq)
#     if not st.session_state["api_key_gemini"]:
#         st.text_input("Enter Groq API Key:", type="password", key="api_key_groq", on_change=update_api_key, args=("groq",))
#         if st.session_state["api_key_groq"]:
#             st.selectbox("Choose Groq Model:", groq_models)



# Sidebar setup for Google Gemini API
with st.sidebar:
    st.header('Google Gemini')
    st.write('### You can get the Google Gemini API key From [Here](https://aistudio.google.com/app/apikey)')
    model_name_gemini = st.selectbox("Choose LLM:", ["gemini-1.5-flash", "gemini-1.5-pro"])
    api_key_gemini = st.text_input('Enter Your API Key: ')
    st.divider()


# If no vectorstore in session, load and process PDFs
if 'vectorstore' not in st.session_state:
    # File uploader
    files = st.sidebar.file_uploader('Upload Your PDFs', type='pdf', accept_multiple_files=True)

    if files:
        docs = []
        
        for file in files:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.read())
                tmp_file_path = tmp_file.name
            
            try:
                # Load PDF using PyPDFLoader
                loader = PyPDFLoader(tmp_file_path)
                docs.extend(loader.load())  # Append extracted docs

            finally:
                # Remove temporary file
                os.remove(tmp_file_path)

        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter()
        chunks = splitter.split_documents(docs)

        # Create embeddings using HuggingFace model
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create FAISS vector store from chunks and embeddings
        st.session_state["vectorstore"] = FAISS.from_documents(chunks, embedding)
        st.success("PDF files processed and vector store created!")
    else:
        st.warning("Please upload PDF files to proceed.")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = [
        {'role': 'assistant', 'content': 'Ask me Any Question about your Document.'}
    ]

# Format document contents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def reset_conversation():
    st.session_state.pop('chat_history')
    st.session_state['chat_history'] = [
        {"role":"assistant", "content":"Ask me Any Question about your Document."}
    ]

def load_chain(question, chat_history, retriver):

    model = ChatGoogleGenerativeAI(api_key=api_key_gemini, model=model_name_gemini, temperature=0.5)
    output_parser = StrOutputParser()

    # Contextualize question chain
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(model, retriver, contextualize_q_prompt)

    system_prompt2 = """
    You are a very helpful assistant. Assitant the user about the docs and answer their questions in very polight way.
    If the user ask you any question which is not in their documents applogies to them and tell them this informtion is not in your docs and offer them is they want you to tell the answer based on your own knowledge.
    Unless they don't ask you. You do not have to answer any question by your self.

    {context}
    """

    prompt2 = ChatPromptTemplate.from_messages(
        [
            ('system', system_prompt2),
            MessagesPlaceholder(variable_name='chat_history'),
            ('human', '{input}')
        ]
    )

    question_answer_chain  = create_stuff_documents_chain(model, prompt2)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain.invoke({"input": question, "chat_history": chat_history})['answer']



# Dividing the main interface into two parts
col1, col2 = st.columns([1, 5])

# Displaying chat history
for message in st.session_state.chat_history:
    avatar = "./images/male.png" if message["role"] == "user" else "./images/scientist.png"
    with col2:
        st.chat_message(message["role"], avatar=avatar).write(message["content"])


with col1:
    st.button("Reset", use_container_width=True, on_click=reset_conversation)


if 'vectorstore' in st.session_state:
    query = st.chat_input("Type your question")
    if query:
        col2.chat_message("user", avatar="./images/male.png").write(query)
        
        st.session_state.chat_history.append({"role": "user", "content": query})

        with col2.chat_message("assistant", avatar="./images/scientist.png"):
            
            response = st.write(load_chain(question=query,
                                retriver=st.session_state["vectorstore"].as_retriever(search_kwargs={"k": 6}),
                                chat_history=st.session_state.chat_history))
        
            # Add response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        