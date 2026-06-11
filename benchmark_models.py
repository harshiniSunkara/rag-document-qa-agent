"""
Benchmarks Mistral, Llama 3, and Qwen across temperature settings.
Resume claim: "experimented with Llama 3 and Qwen across temperature/top-p 
settings to inform model selection"
"""

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import ChatPromptTemplate
import json

CHROMA_PATH = "chroma"

MODELS = ["mistral", "llama3", "qwen2.5"]
TEMPERATURES = [0.0, 0.5, 1.0]

TEST_QUESTIONS = [
    ("What does Alice drink to make herself small?",   ["drink me", "bottle"]),
    ("What does the Caterpillar sit on?",              ["mushroom"]),
    ("What does the Queen use as croquet mallets?",    ["flamingo"]),
    ("What is at the bottom of the well?",             ["cupboard", "bookshelf", "maps"]),
    ("Who is the Knave accused of stealing?",          ["tarts"]),
]

PROMPT = """Answer using ONLY the context. Be concise.
Context: {context}
Question: {question}
Answer:"""

def get_context(question, db):
    results = db.similarity_search_with_relevance_scores(question, k=5)
    if not results or results[0][1] < 0.1:
        return ""
    return "\n---\n".join([doc.page_content for doc, _ in results])

def check(answer, keywords):
    return any(k.lower() in answer.lower() for k in keywords)

def main():
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = {}
    prompt_template = ChatPromptTemplate.from_template(PROMPT)

    for model_name in MODELS:
        print(f"\n{'='*50}")
        print(f"MODEL: {model_name}")
        print('='*50)
        results[model_name] = {}

        for temp in TEMPERATURES:
            model = ChatOllama(model=model_name, temperature=temp)
            correct = 0

            for question, keywords in TEST_QUESTIONS:
                context = get_context(question, db)
                chain = prompt_template | model
                answer = chain.invoke({"context": context, "question": question}).content.strip()
                passed = check(answer, keywords)
                if passed:
                    correct += 1
                print(f"  temp={temp} | {'✓' if passed else '✗'} | {question[:50]}")

            score = round(correct / len(TEST_QUESTIONS) * 100, 1)
            results[model_name][f"temp_{temp}"] = score
            print(f"  >> temp={temp} score: {score}%")

    # Summary table
    print(f"\n{'='*50}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*50}")
    print(f"{'Model':<12} {'temp=0':>8} {'temp=0.5':>10} {'temp=1.0':>10}")
    print("-" * 42)
    for model_name in MODELS:
        r = results[model_name]
        print(f"{model_name:<12} {r['temp_0.0']:>7}% {r['temp_0.5']:>9}% {r['temp_1.0']:>9}%")

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to benchmark_results.json")

if __name__ == "__main__":
    main()