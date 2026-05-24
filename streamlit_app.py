import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


# TITLE
st.title("steve jobs RAG assistant")


# EMBEDDING MODEL
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


# LOAD VECTOR DB
vectorstore = Chroma(
    persist_directory="./db",
    embedding_function=embedding_model
)


# RETRIEVER
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# LLM
llm = ChatOllama(
    model="llama3",
    temperature=0
)



query = st.text_input("Ask a question about Steve Jobs..........")


# QA
if query:

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
    Answer only from the context.

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)

    st.write(response.content)