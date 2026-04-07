from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain.schema import Document
import os

def conversational_rag(documents: list) -> list:
    """Returns answers for a 2-turn RAG conversation."""
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    docs = [
        Document(page_content=d["content"], metadata=d.get("metadata", {}))
        for d in documents
    ]
    
    vectorstore = PGVector.from_documents(
        documents=docs,
        embedding=embeddings,
        connection_string=os.getenv("PG_CONNECTION_STRING"),
        collection_name="rag_chat"
    )
    
    retriever = vectorstore.as_retriever()
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        retriever=retriever
    )
    
    chat_history = []
    
    # Q1
    q1 = "What is AI?"
    res1 = qa_chain({"question": q1, "chat_history": chat_history})
    chat_history.append((q1, res1["answer"]))
    
    # Q2
    q2 = "Can you summarize it?"
    res2 = qa_chain({"question": q2, "chat_history": chat_history})
    
    return [res1["answer"], res2["answer"]]