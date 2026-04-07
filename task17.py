from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
import os

def rag_agent(question: str) -> str:
    """Uses a ReAct agent with a retriever tool to answer the question."""
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = PGVector(
        connection_string=os.getenv("PG_CONNECTION_STRING"),
        embedding=embeddings,
        collection_name="rag_agent"
    )
    
    retriever = vectorstore.as_retriever()
    
    def retrieve_docs(q):
        docs = retriever.get_relevant_documents(q)
        return "\n".join([doc.page_content for doc in docs])
    
    tools = [
        Tool(
            name="Retriever",
            func=retrieve_docs,
            description="Fetch relevant documents"
        )
    ]
    
    agent = initialize_agent(
        tools,
        ChatOpenAI(model="gpt-4o-mini"),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    return agent.run(question)