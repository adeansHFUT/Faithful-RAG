import asyncio
from datasets import load_dataset, Dataset
import numpy as np
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ.setdefault("NLTK_DATA", "/share/home/yangjj/nltk_data")

import nltk
nltk.data.path.append("/share/home/yangjj/nltk_data")

from faithfulrag import FaithfulRAG  

async def main():
    # 1. Load dataset
    dataset = load_dataset("json", data_files="./datas/faitheval_data.json")
    # dataset = dataset['train'].select(range(10))
    dataset = dataset['train']

    # 2. Initialize FaithfulRAG pipeline
    rag = FaithfulRAG(
        backend_type="hf",
        model_name="/share/home/yangjj/models/qwen2.5-7b-instruct",
        similarity_model="/share/home/yangjj/models/bge-large-en-v1.5",
        knowledges_cache_path="cache/knowledges.jsonl",
        contexts_cache_path="cache/self_contexts.jsonl",
        self_facts_cache_path="cache/self_facts.jsonl",
        chunks_cache_path="cache/topk_chunks.jsonl",
        predictions_cache_path="cache/preds.jsonl",
    )
    
    # 3. Generate self-consistent facts
    print("Generating self-consistent facts...")
    self_facts = await rag.get_self_facts(
        dataset,
        fact_mining_type="default",
        temperature=0.0  # Override default parameter
    )
    print(f"Generated facts sample: {self_facts[0]['facts'][0]}\n")
    
    # 4. Retrieve top-k contextual chunks
    print("Retrieving contextual chunks...")
    topk_chunks = rag.get_topk_chunks(
        dataset,
        self_facts,
    )
    print(f"Top chunks sample: {topk_chunks[0]['topk_chunks'][0]}\n")
    
    # 5. Generate predictions
    print("Generating predictions...")
    predictions = await rag.get_predictions(
        dataset,
        topk_chunks,
        generation_type="wo_cot",  # Try "scheduled_cot" or "wo_cot"
        max_tokens=500,  # Override generation parameter,
        temperature=0.0
    )
    print(f"Predictions: {predictions}\n")
    
    # 6. Evaluate results
    print("Evaluating predictions...")
    evaluation = rag.evaluate(
        dataset,
        predictions,
        detailed_output=True
    )
    
    print("\nEvaluation Results:")
    print(f"Exact Match: {evaluation['exact_match']:.2f}%")
    print(f"Accuracy: {evaluation['acc']:.2f}%")
    print(f"F1 Score: {evaluation['f1']:.2f}%")

    os.makedirs("results", exist_ok=True)
    
    with open("results/evaluation_results.json", "w") as f:
        json.dump(evaluation, f, indent=4)

if __name__ == "__main__":
    asyncio.run(main())