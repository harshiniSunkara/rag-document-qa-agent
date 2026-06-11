"""
Evaluation harness: compares RAG pipeline vs base LLM (no context)
on factuality and hallucination across 20+ test cases.

Resume claim: "Authored 20+ eval test cases benchmarking RAG vs. base LLM
on factuality and hallucination"
"""

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import ChatPromptTemplate
import json
import time

CHROMA_PATH = "chroma"
MODEL_NAME  = "mistral"

# ── 20 TEST CASES ─────────────────────────────────────────────────────────────
# Format: question, expected keywords that a correct answer should contain,
# and a hallucination_risk flag (True = base LLM likely to hallucinate)
TEST_CASES = [
    {
        "question": "Who is the author of Alice in Wonderland?",
        "expected_keywords": ["lewis carroll", "carroll"],
        "hallucination_risk": False
    },
    {
        "question": "What does Alice follow into the rabbit hole?",
        "expected_keywords": ["rabbit", "white rabbit", "hatter", "late"],
        "hallucination_risk": False
    },
    {
        "question": "What does Alice drink to make herself small?",
        "expected_keywords": ["bottle", "drink me", "liquid"],
        "hallucination_risk": True
    },
    {
        "question": "What is the name of the cat in the story?",
        "expected_keywords": ["cheshire", "cheshire cat"],
        "hallucination_risk": False
    },
    {
        "question": "Who is always late in the story?",
        "expected_keywords": ["white rabbit", "rabbit"],
        "hallucination_risk": False
    },
    {
        "question": "What game do they play at the Queen's croquet ground?",
        "expected_keywords": ["croquet"],
        "hallucination_risk": True
    },
    {
        "question": "Who shouts off with their head in the story?",
        "expected_keywords": ["queen", "queen of hearts"],
        "hallucination_risk": False
    },
    {
        "question": "What does the Caterpillar sit on?",
        "expected_keywords": ["mushroom"],
        "hallucination_risk": True
    },
    {
        "question": "What is the Mad Hatter always doing?",
        "expected_keywords": ["tea", "tea party", "tea-party", "time", "6", "six"],
        "hallucination_risk": False
    },
    {
        "question": "How does Alice enter Wonderland?",
        "expected_keywords": ["rabbit hole", "falls", "hole"],
        "hallucination_risk": False
    },
    {
        "question": "What does eating the mushroom do to Alice?",
        "expected_keywords": ["grow", "shrink", "size", "larger", "smaller"],
        "hallucination_risk": True
    },
    {
        "question": "Who is the Mock Turtle?",
        "expected_keywords": ["turtle", "mock turtle", "sad"],
        "hallucination_risk": True
    },
    {
        "question": "What trial takes place at the end of the story?",
        "expected_keywords": ["tarts", "stolen", "knave", "trial"],
        "hallucination_risk": True
    },
    {
        "question": "What does the Cheshire Cat slowly disappear into?",
        "expected_keywords": ["grin", "smile", "air"],
        "hallucination_risk": True
    },
    {
        "question": "Who does Alice meet at the tea party?",
        "expected_keywords": ["hatter", "mad hatter", "march hare", "dormouse"],
        "hallucination_risk": False
    },
    {
        "question": "What is written on the bottle Alice finds?",
        "expected_keywords": ["drink me"],
        "hallucination_risk": False
    },
    {
    "question": "What does Alice eat to make herself grow larger?",
    "expected_keywords": ["cake", "eat me", "mushroom", "larger"],
    "hallucination_risk": True
    },
    {
        "question": "What does the Queen use as croquet mallets?",
        "expected_keywords": ["flamingo", "flamingos"],
        "hallucination_risk": True
    },
    {
        "question": "Who are the cards painting roses in the garden?",
        "expected_keywords": ["cards", "playing cards", "red", "roses"],
        "hallucination_risk": True
    },
    {
        "question": "What is at the bottom of the well Alice falls into?",
        "expected_keywords": ["cupboards", "bookshelves", "maps", "pictures"],
        "hallucination_risk": True
    },
    {
        "question": "What does the Duchess's baby turn into?",
        "expected_keywords": ["pig"],
        "hallucination_risk": True
    },
    {
        "question": "Who is the Knave accused of stealing?",
        "expected_keywords": ["tarts", "queen", "hearts"],
        "hallucination_risk": True
    },
]

RAG_PROMPT = """Answer the question using ONLY the context below. Be concise.

Context:
{context}

Question: {question}
Answer:"""

BASE_PROMPT = """Answer the following question about Alice in Wonderland. Be concise.

Question: {question}
Answer:"""


def check_answer(answer: str, expected_keywords: list) -> bool:
    """Returns True if any expected keyword appears in the answer."""
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in expected_keywords)


def run_rag(question: str, db, model) -> str:
    """Run the full RAG pipeline and return the answer."""
    results = db.similarity_search_with_relevance_scores(question, k=5)
    if not results or results[0][1] < 0.1:
        return "No relevant context found."
    context = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    chain = prompt | model
    return chain.invoke({"context": context, "question": question}).content.strip()


def run_base(question: str, model) -> str:
    """Run the base LLM with NO context (tests hallucination)."""
    prompt = ChatPromptTemplate.from_template(BASE_PROMPT)
    chain = prompt | model
    return chain.invoke({"question": question}).content.strip()


def main():
    print("=" * 60)
    print("RAG vs BASE LLM EVALUATION HARNESS")
    print(f"Model: {MODEL_NAME}  |  Test cases: {len(TEST_CASES)}")
    print("=" * 60)

    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=embedding_function)
    model = ChatOllama(model=MODEL_NAME, temperature=0)

    rag_correct   = 0
    base_correct  = 0
    rag_results   = []

    for i, test in enumerate(TEST_CASES):
        q        = test["question"]
        expected = test["expected_keywords"]
        risk     = test["hallucination_risk"]

        print(f"\n[{i+1}/{len(TEST_CASES)}] {q}")

        rag_answer  = run_rag(q, db, model)
        base_answer = run_base(q, model)

        rag_pass  = check_answer(rag_answer,  expected)
        base_pass = check_answer(base_answer, expected)

        if rag_pass:  rag_correct  += 1
        if base_pass: base_correct += 1

        status_rag  = "✓ PASS" if rag_pass  else "✗ FAIL"
        status_base = "✓ PASS" if base_pass else "✗ FAIL"

        print(f"  RAG  [{status_rag}]  → {rag_answer[:120]}")
        print(f"  BASE [{status_base}] → {base_answer[:120]}")

        rag_results.append({
            "question":        q,
            "expected":        expected,
            "hallucination_risk": risk,
            "rag_answer":      rag_answer,
            "base_answer":     base_answer,
            "rag_correct":     rag_pass,
            "base_correct":    base_pass,
        })

        time.sleep(1)   # be gentle with local CPU

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    total        = len(TEST_CASES)
    rag_score    = round(rag_correct  / total * 100, 1)
    base_score   = round(base_correct / total * 100, 1)
    improvement  = round(rag_score - base_score, 1)

    # Count hallucination-risk cases only
    risk_cases      = [r for r in rag_results if r["hallucination_risk"]]
    rag_risk_pass   = sum(1 for r in risk_cases if r["rag_correct"])
    base_risk_pass  = sum(1 for r in risk_cases if r["base_correct"])
    halluc_reduction = round(
        (1 - base_risk_pass / max(len(risk_cases), 1)) * 100 -
        (1 - rag_risk_pass  / max(len(risk_cases), 1)) * 100, 1
    )

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"RAG  accuracy : {rag_correct}/{total}  ({rag_score}%)")
    print(f"BASE accuracy : {base_correct}/{total}  ({base_score}%)")
    print(f"Overall improvement : +{improvement}%")
    print(f"Hallucination-risk cases : {len(risk_cases)}")
    print(f"  RAG  correct on risk cases : {rag_risk_pass}/{len(risk_cases)}")
    print(f"  BASE correct on risk cases : {base_risk_pass}/{len(risk_cases)}")
    print(f"Hallucination reduction     : {halluc_reduction}%")
    print("=" * 60)

    # Save results to JSON for your GitHub README
    with open("eval_results.json", "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "total_tests": total,
            "rag_accuracy": rag_score,
            "base_accuracy": base_score,
            "improvement": improvement,
            "hallucination_reduction": halluc_reduction,
            "details": rag_results
        }, f, indent=2)

    print("\nFull results saved to eval_results.json")


if __name__ == "__main__":
    main()