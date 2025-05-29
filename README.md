# Faithful-RAG
This repo contains the code and data for **FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful Retrieval-Augmented Generation(ACL25)**. A novel RAG model (FaithfulRAG) that achieves context faithfulness while maintaining accurate knowledge integration.

![image](./pipeline.jpg)


## Usage
### 1. Dependencies
Run the following command to install the dependencies:
```shell
pip install -r requirements.txt
```
### 2. Quick Start
```python
import asyncio
from datasets import Dataset
from faithfulrag import FaithfulRAG

# Load dataset
dataset = load_dataset("json", data_files="./datas/faitheval_data.json")
dataset = dataset['train']

# Initialize FaithfulRAG pipeline
rag = FaithfulRAG(
    backend_type="openai",  # or "hf", "llamafactory"
    model_name="gpt-3.5-turbo",
    similarity_model="all-MiniLM-L6-v2"
)

async def run_pipeline():
    # Generate self-consistent facts
    self_facts = await rag.get_self_facts(dataset)
    
    # Retrieve top contextual chunks
    topk_chunks = rag.get_topk_chunks(dataset, self_facts)
    
    # Generate predictions
    predictions = await rag.get_predictions(
        dataset, 
        topk_chunks,
        generation_type="normal_cot"  # or "scheduled_cot", "wo_cot"
    )
    
    # Evaluate results
    results = rag.evaluate(dataset, predictions, cot_format=True)
    print(f"Exact Match: {results['exact_match']:.2f}%")

asyncio.run(run_pipeline())
```
