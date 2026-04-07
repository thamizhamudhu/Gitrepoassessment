from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langsmith import Client
import uuid

def traced_chain(topic: str) -> dict:
    """Runs a chain with langsmith tracing. Returns answer and run id."""
    
    llm = ChatOpenAI(model="gpt-4o-mini")
    client = Client()
    
    chain = RunnableLambda(lambda x: f"Explain {x}") | llm
    
    run_name = f"trace-{uuid.uuid4()}"
    
    result = chain.invoke(
        topic,
        config={
            "run_name": run_name,
            "tags": ["task18"]
        }
    )
    
    return {
        "question": topic,
        "answer": result.content,
        "run_name": run_name
    }