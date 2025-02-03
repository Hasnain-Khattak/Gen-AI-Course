import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from typing import List
from langchain.load import dumps, loads
from langchain_core.output_parsers import BaseOutputParser

# Output parser will split the LLM result into a list of queries
class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Remove empty lines

output_parser = LineListOutputParser()


api_key = st.session_state.get("api_key_gemini")
model_name = st.session_state.get("model_name_gemini")

model = ChatGoogleGenerativeAI(model=model_name, api_key=api_key)


# Multi Query: Different Perspectives
template = """You are an AI language model assistant. Your task is to generate
three different versions of the given user question to retrieve relevant
documents from a vector database. By generating multiple perspectives on the user question,
your goal is to help the user overcome some of the limitations of the distance-based similarity search. 
Provide three alternative questions separated by newlines.
Original question: 
{question}"""

prompt_perspectives = ChatPromptTemplate.from_template(template)

retriever = st.session_state["vectorstore"].as_retriever()

def get_unique_union(documents: list[list]):
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))

    return [loads(doc) for doc in unique_docs]


generate_queries = (
    prompt_perspectives
    | model
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)
retrieval_chain = generate_queries | retriever.map() | get_unique_union



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def multi_query_rag_response(question):
    prompt = hub.pull("rlm/rag-prompt")

    rag_chain = (
        {"context": retrieval_chain | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return rag_chain.invoke(question)


if st.session_state.messages[-1]["role"] == "user":
    with st.spinner("Thinking..."):
        response = multi_query_rag_response(st.session_state.messages[-1]["content"])
        print("\n\nMulti-Query retrieval_chain: ", retrieval_chain.invoke(st.session_state.messages[-1]["content"]))
        st.chat_message("assistant", avatar="images/scientist.png").markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})