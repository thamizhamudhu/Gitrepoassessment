# run_evaluation.py

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def run_langsmith_evaluation():
    # placeholder evaluation result
    return {"score": 0.9}


if __name__ == "__main__":
    result = run_langsmith_evaluation()

    print("Evaluation Result:")
    print(result)