import argparse
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain import hub
import json

CHROMA_PATH = "chroma"
MODEL_NAME = "mistral"

# ── 1. JSON OUTPUT SCHEMA ────────────────────────────────────────────────────
# This tells the LLM exactly what structure to return
# In an interview: "I used Pydantic to define a strict output schema so the 
# response could be parsed and consumed programmatically downstream"
class RAGResponse(BaseModel):
    answer: str = Field(description="The answer to the question")
    confidence: str = Field(description="high, medium, or low")
    sources: list = Field(description="List of source documents used")

# ── 2. FEW-SHOT PROMPT TEMPLATE ──────────────────────────────────────────────
# Few-shot = giving the LLM examples of good behaviour before asking it to respond
# This teaches it the format and style we want
few_shot_examples = [
    {
        "question": "Who is the main character?",
        "context": "Alice was beginning to get very tired of sitting by her sister...",
        "answer": "The main character is Alice, a young girl who falls down a rabbit hole."
    },
    {
        "question": "Where does the story take place?",
        "context": "...she had peeped into the book her sister was reading... a White Rabbit ran close by her.",
        "answer": "The story begins in a meadow and then moves to Wonderland, a fantastical underground world."
    }
]

example_prompt = PromptTemplate(
    input_variables=["question", "context", "answer"],
    template="Question: {question}\nContext: {context}\nAnswer: {answer}"
)

few_shot_prompt = FewShotPromptTemplate(
    examples=few_shot_examples,
    example_prompt=example_prompt,
    prefix="You are an expert on the document. Answer questions using ONLY the context provided. Here are some examples:\n",
    suffix="\nNow answer this:\nQuestion: {question}\nContext: {context}\nAnswer:",
    input_variables=["question", "context"]
)

# ── 3. CHAIN-OF-THOUGHT PROMPT ───────────────────────────────────────────────
# Chain-of-thought = asking the LLM to reason step by step before answering
# Reduces hallucination by forcing explicit reasoning
COT_PROMPT_TEMPLATE = """
You are an expert on the document. Use the following context to answer the question.

Think step by step:
1. What is the question asking for?
2. What relevant information exists in the context?
3. What is the most accurate answer based only on the context?

Context:
{context}

---

Question: {question}

Let's think step by step before answering:
"""

def get_retriever_tool(db):
    """
    Wraps our ChromaDB retriever as a LangChain Tool.
    The ReAct agent will call this when it decides the answer is in the document.
    """
    def retrieve(query: str) -> str:
        results = db.similarity_search_with_relevance_scores(query, k=3)
        if not results or results[0][1] < 0.3:
            return "No relevant results found in the document."
        context = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
        sources = [doc.metadata.get("source", "unknown") for doc, _ in results]
        return f"CONTEXT:\n{context}\n\nSOURCES:\n{sources}"

    return Tool(
        name="DocumentRetriever",
        func=retrieve,
        description=(
            "Use this tool to search the Alice in Wonderland document for answers. "
            "Input should be a search query. Use this FIRST before web search."
        )
    )

def query_rag(query_text: str, use_agent: bool = False, use_cot: bool = False):
    # Load the vector DB with the same embedding model used to create it
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Load the LLM
    model = ChatOllama(model=MODEL_NAME, temperature=0.1)

    if use_agent:
        # ── REACT AGENT MODE ─────────────────────────────────────────────────
        # ReAct = Reasoning + Acting loop
        # The agent sees the question, decides which tool to use (document or web),
        # observes the result, then decides next action until it has an answer
        print(f"\n[MODE: ReAct Agent with tools]\n")

        retriever_tool = get_retriever_tool(db)
        search_tool = DuckDuckGoSearchRun()
        web_tool = Tool(
            name="WebSearch",
            func=search_tool.run,
            description=(
                "Use this tool ONLY if the document retriever returns no relevant results. "
                "Searches the web for current information."
            )
        )

        tools = [retriever_tool, web_tool]

        # Pull the standard ReAct prompt from LangChain hub
        react_prompt = hub.pull("hwchase17/react")

        agent = create_react_agent(model, tools, react_prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,       # Shows the Thought/Action/Observation loop
            handle_parsing_errors=True,
            max_iterations=5
        )

        result = agent_executor.invoke({"input": query_text})
        print(f"\nFinal Answer: {result['output']}")
        return result['output']

    else:
        # ── STANDARD RAG MODE (with few-shot or CoT) ─────────────────────────
        results = db.similarity_search_with_relevance_scores(query_text, k=3)

        if not results or results[0][1] < 0.3:
            print("Unable to find matching results.")
            return

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
        sources = [doc.metadata.get("source", "unknown") for doc, _ in results]

        if use_cot:
            # Chain-of-thought mode
            print(f"\n[MODE: Chain-of-Thought RAG]\n")
            prompt_template = ChatPromptTemplate.from_template(COT_PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=query_text)
        else:
            # Few-shot mode
            print(f"\n[MODE: Few-Shot RAG with JSON output]\n")
            prompt = few_shot_prompt.format(question=query_text, context=context_text)

        response_text = model.invoke(prompt).content

        # ── JSON OUTPUT PARSER ────────────────────────────────────────────────
        # Try to return a structured JSON response
        # In an interview: "I used structured output parsing to make responses
        # machine-readable for downstream consumption"
        parser = JsonOutputParser(pydantic_object=RAGResponse)
        json_prompt = f"""
Based on this answer, return a JSON object with exactly these fields:
- "answer": the answer text
- "confidence": "high", "medium", or "low" based on how well the context supported the answer  
- "sources": {sources}

Answer to format: {response_text}

Return ONLY valid JSON, nothing else.
"""
        try:
            json_response = model.invoke(json_prompt).content
            # Clean up common LLM JSON formatting issues
            json_response = json_response.strip()
            if json_response.startswith("```"):
                json_response = json_response.split("```")[1]
                if json_response.startswith("json"):
                    json_response = json_response[4:]
            parsed = json.loads(json_response)
            print(f"Response: {parsed['answer']}")
            print(f"Confidence: {parsed['confidence']}")
            print(f"Sources: {parsed['sources']}")
            return parsed
        except Exception:
            # Fallback to plain text if JSON parsing fails
            print(f"Response: {response_text}")
            print(f"Sources: {sources}")
            return response_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--agent", action="store_true", help="Use ReAct agent mode")
    parser.add_argument("--cot", action="store_true", help="Use chain-of-thought mode")
    args = parser.parse_args()

    query_rag(args.query_text, use_agent=args.agent, use_cot=args.cot)

if __name__ == "__main__":
    main()