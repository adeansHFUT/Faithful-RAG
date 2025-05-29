# Faithful-RAG
A novel RAG model (FaithfulRAG) that achieves context faithfulness while maintaining accurate knowledge integration.

![image](./pipeline.jpg)


## Usage
### 1. Dependencies
Run the following command to install the dependencies:
```shell
pip install -r requirements.txt
```
### 2. Prepare Data
The datasets used in our paper are located under ./data. If you want to use your own dataset, you need to convert the dataset labels into the following format:
```json
{
    "id": "1",
    "question": "question",
    "context": "context",
    "answer": "ground truth",
}
```
### 3. Quick Start
Import the dependencies:
```python
from faithful_rag.pipeline import FaithfulRAG
from datasets import load_dataset
```
Load the dataset and initialize Faithful-RAG:
```python
dataset = load_dataset("your test dataset", split="test")
faithful_rag = FaithfulRAG(generator_llm="Meta-Llama-3.1-8B-Instruct",
                           generator_llm_type='llama3',
                           similarity_model='all-MiniLM-L6-v2'
)
```


result = faithful_rag.evaluate(dataset, predictions)
Get self-facts:
```python
self_facts = faithful_rag.get_self_facts(dataset)

```
Get contextual_chunks
```python
contextual_chunks = faithful_rag.get_contextual_chunks(dataset, self_facts)
```
Predict answers:
```python
predictions = faithful_rag.get_predictions(dataset, contextual_chunks)
```
Evaluate results:
```python
predictions = faithful_rag.evaluate(dataset, predictions)
```
