import streamlit as st
from langchain_cohere import CohereRerank
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

Cohereapi_key = st.sidebar.text_input('Enter Your Cohere API Key:', value=st.session_state.cohere_api, type="password")
if Cohereapi_key and Cohereapi_key != st.session_state.cohere_api:
        st.session_state.cohere_api = Cohereapi_key
        
api_key = st.session_state.get("api_key_gemini")
model_name = st.session_state.get("model_name_gemini")
cohere_api_key = st.session_state.get('cohere_api')


retriever = st.session_state["vectorstore"].as_retriever(search_type="similarity", search_kwargs={"k": 6})

compressor = CohereRerank(model="rerank-english-v3.0", top_n=6, cohere_api_key=cohere_api_key)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

model = ChatGoogleGenerativeAI(model=model_name, api_key=api_key)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def reranker_rag(question):
    prompt = hub.pull("rlm/rag-prompt")
    parser = StrOutputParser()
    reranked_rag_chain = (
        {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | parser
    )
    return reranked_rag_chain.invoke(question)

if st.session_state.messages[-1]["role"] == "user":
    with st.spinner("Thinking..."):
        response = reranker_rag(st.session_state.messages[-1]["content"])
        print("\n\ncompression_retriever: ", compression_retriever.invoke(st.session_state.messages[-1]["content"]))
        st.chat_message("assistant", avatar="images/scientist.png").markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})