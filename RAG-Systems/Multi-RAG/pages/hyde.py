import streamlit as st
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub


api_key = st.session_state.get("api_key_gemini")
model_name = st.session_state.get("model_name_gemini")

retriever = st.session_state["vectorstore"].as_retriever(
    search_type="similarity", search_kwargs={"k": 6}
)

template = "For the given question, generate a hypothetical answer and nothing else.\n\n{question}"
prompt = ChatPromptTemplate.from_template(template)
parser =  StrOutputParser()

model = ChatGoogleGenerativeAI(model=model_name, api_key=api_key)


first_chain = prompt | model | parser

retrieval_chain = first_chain | retriever


def format_docs(docs):
    """Formats retrieved documents into a single string for better processing."""
    return "\n\n".join(doc.page_content for doc in docs)


def hyde_rag(question):
    """Handles the RAG response generation process."""
    try:
        prompt = hub.pull("rlm/rag-prompt")

        rag_chain = (
            {"context": retrieval_chain | format_docs, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )

        return rag_chain.invoke(question)

    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "Sorry, an error occurred while processing your request."



# Process user query if the last message is from the user
if st.session_state.messages[-1]["role"] == "user":
    with st.spinner("Thinking..."):
        response = hyde_rag(st.session_state.messages[-1]["content"])
        print("\n\nHyDe retrieval_chain: ", retrieval_chain.invoke(st.session_state.messages[-1]["content"]))
        st.chat_message("assistant", avatar="images/scientist.png").markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
