# RAG Document Q&A Agent with Open-Source LLMs

End-to-end Retrieval-Augmented Generation (RAG) pipeline using LangChain, ChromaDB, and local open-source LLMs via Ollama. Built to demonstrate core LLM engineering primitives: chunking, embeddings, vector retrieval, ReAct agents, prompt engineering, and LLM evaluation.

## Features
- **RAG Pipeline**: PDF ingestion → chunking → nomic-embed-text embeddings → ChromaDB vector store → Mistral 7B generation
- **ReAct Agent**: Tool-using agent that dynamically chooses between document retrieval and DuckDuckGo web search
- **Prompt Engineering**: Few-shot prompting, chain-of-thought reasoning, and structured JSON output parsing
- **Evaluation Harness**: 22 test cases benchmarking RAG vs base LLM on factuality and hallucination
- **Model Benchmarking**: Mistral 7B vs Llama 3 vs Qwen 2.5 across temperature=0, 0.5, 1.0

## Benchmark Results

| Model    | temp=0 | temp=0.5 | temp=1.0 |
|----------|--------|----------|----------|
| Mistral  | 80.0%  | 80.0%    | 80.0%    |
| Llama 3  | 80.0%  | 80.0%    | 80.0%    |
| Qwen 2.5 | 60.0%  | 60.0%    | 60.0%    |

**Finding**: Mistral selected as production model — equal accuracy to Llama 3, smaller footprint (4.4GB vs 4.7GB), faster on CPU. Temperature had no meaningful effect on factual retrieval tasks, confirming low temperature (0.0) is optimal for RAG.

## Eval Results (RAG vs Base LLM)
- RAG accuracy: 90.9% (20/22 test cases)
- Base LLM accuracy: 95.5% (21/22 test cases)  
- Key finding: RAG advantage is strongest on domain-specific/private documents. On well-known texts, base LLM knowledge competes. RAG is essential when documents are proprietary or post-training-cutoff.

## Stack
Python · LangChain · ChromaDB · Mistral 7B · Llama 3 · Qwen 2.5 · Ollama · nomic-embed-text · DuckDuckGo · Git

## Setup
```bash
pip install -r requirements.txt
ollama pull mistral && ollama pull nomic-embed-text
python create_database.py   # builds vector store
python query_data.py "your question"              # few-shot + JSON output
python query_data.py "your question" --cot        # chain-of-thought
python query_data.py "your question" --agent      # ReAct agent
python test_rag.py                                # run eval harness
python benchmark_models.py                        # model comparison
```
