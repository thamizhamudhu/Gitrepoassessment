# Gitrepoassessment

langchain-rag-tasks

This repo contains my implementation for some selected tasks from the LangChain + RAG challenge.

I mainly focused on embeddings, RAG, and LangSmith parts because they felt more important for understanding real-world systems.

tasks covered
task 6 → cosine similarity (manual + numpy)
task 7 → chunking + batch embeddings
task 8 → comparing embedding models
task 14 → basic rag
task 15 → rag with sources
task 16 → conversational rag
task 17 → rag agent
task 18 → tracing with langsmith
task 19 → dataset creation
task 20 → evaluation
what i understood
cosine similarity is basically how close two vectors are in direction, not magnitude
chunking is needed because models have token limits and also improves retrieval
embeddings from different models have different dimensions
rag = retrieve + generate (simple idea but implementation has many small steps)
adding sources helps to verify answers
conversational rag needs memory otherwise it breaks context
agents are slightly unpredictable but powerful
langsmith helps to actually see what is going on inside chains
how to run

install requirements:

pip install -r requirements.txt

add .env file:

OPENAI_API_KEY=xxx
LANGCHAIN_API_KEY=xxx

run:

python main.py
notes
some parts may not be fully optimized
pgvector not included in these tasks
still exploring better chunking strategies
agent responses can vary sometimes
