import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title='PY Professor', page_icon='🧠')

# Title
st.markdown("""
    <h1 style='text-align: center; font-size: 4em; font-weight: bold;'>
        <span style='color: #fcfcfc;'>Python</span> 
        <span style='color: #00c220;'>Professor</span>
    </h1>
""", unsafe_allow_html=True)


# Sidebar setup for Google Gemini API
with st.sidebar:
    st.header('Google Gemini')
    st.write('### You can get the Google Gemini API key From [Here](https://aistudio.google.com/app/apikey)')
    model_name_gemini = st.selectbox("Choose LLM:", ["gemini-1.5-flash", "gemini-1.5-pro"])
    api_key_gemini = st.text_input('Enter Your API Key: ')
    st.divider()

# Gender Selection Section
if 'gender' not in st.session_state:
    # Ask the user to select their gender first
    st.title("Welcome to Python Professor!")
    st.write("Please select your gender to start:")
    gender = st.sidebar.radio("Choose your gender", ("Male", "Female"))

    if gender:
        st.session_state['gender'] = gender  # Store gender in session state
        st.rerun()  # Reload the page to show chat interface after gender selection

# Gender-specific Avatars
male_avatar = "./images/male.png"
female_avatar = "./images/female.png"
assistant_avatar = "./images/scientist.png"  # Common avatar for assistant



# If no vectorstore in session, load it
if 'vectorstore' not in st.session_state:
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    st.session_state["vectorstore"] = FAISS.load_local('./FAISS', embedding, allow_dangerous_deserialization=True)

# Initialize chat history if not present
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = [
        {"role": "assistant", "content": "How can I help you today in your programming journey?"}
    ]

# Format document contents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Reset conversation
def reset_conversation():
    st.session_state.pop('chat_history')
    st.session_state['chat_history'] = [
        {"role": "assistant", "content": "How can I help you today in your programming journey?"}
    ]

# Chatbot response chain function
def chain(question, chat_history, retriver):
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

    first_chain = contextualize_q_prompt | model | output_parser
    system_prompt2 = """
    You are a very Professional Developer Teacher, Consultant of Python.
    You have 20+ years of experience in creating various types of projects, specifically using Python.
    You also have 10+ years of experience in teaching the Python programming language.
    Your job is to help students in solving their doubts in any type of project related to Python programming.
    You have to listen to their problems and try to give them short and accurate solutions based on the context.

    {context}
    """

    prompt2 = ChatPromptTemplate.from_messages(
        [
            ('system', system_prompt2),
            MessagesPlaceholder(variable_name='chat_history'),
            ('human', '{input}')
        ]
    )

    rag_chain = (
        RunnablePassthrough.assign(
            context=first_chain | retriver | format_docs
        )
        | prompt2
        | model
        | output_parser
    )

    return rag_chain.stream({"input": question, "chat_history": chat_history})



# Check if gender is selected, if so proceed with the chat interface
if 'gender' in st.session_state:
    # Display the chat interface
    col1, col2 = st.columns([1, 5])

    # Displaying chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            avatar = male_avatar if st.session_state['gender'] == "Male" else female_avatar
        else:  # For the assistant's messages
            avatar = assistant_avatar

    with col2:
        st.chat_message(message["role"], avatar=avatar).write(message["content"])


    with col1:
        st.button("Reset", use_container_width=True, on_click=reset_conversation)

    query = st.chat_input('Enter your message')

    if query:
        col2.chat_message('user', avatar=male_avatar if st.session_state['gender'] == "Male" else female_avatar).write(query)
        st.session_state.chat_history.append({'role': 'user', 'content': query})
        # Generate response
        with col2.chat_message("assistant", avatar=assistant_avatar):
            response = st.write_stream(chain(question=query, retriver=st.session_state["vectorstore"].as_retriever(search_kwargs={"k": 6}),
                                chat_history=st.session_state.chat_history))

            # Add response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
