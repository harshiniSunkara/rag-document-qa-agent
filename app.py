import os
import json
import shutil
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
try:
    from langchain import hub
    HUB_AVAILABLE = True
except Exception:
    HUB_AVAILABLE = False

app = Flask(__name__)
app.secret_key = "rag-app-secret-key"

UPLOAD_FOLDER = "uploads"
CHROMA_PATH   = "chroma"
ALLOWED_EXTENSIONS = {"pdf", "txt", "md"}
MODEL_NAME = "mistral"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Prompt templates ──────────────────────────────────────────────────────────
FEW_SHOT_EXAMPLES = [
    {"question": "Who is the main character?",
     "context": "Alice was beginning to get very tired...",
     "answer": "The main character is Alice."},
    {"question": "Where does the story take place?",
     "context": "...she had peeped into the book her sister was reading...",
     "answer": "The story takes place in Wonderland."},
]
example_prompt = PromptTemplate(
    input_variables=["question", "context", "answer"],
    template="Question: {question}\nContext: {context}\nAnswer: {answer}"
)
few_shot_prompt = FewShotPromptTemplate(
    examples=FEW_SHOT_EXAMPLES,
    example_prompt=example_prompt,
    prefix="You are a document expert. Answer using ONLY the context. Examples:\n",
    suffix="\nQuestion: {question}\nContext: {context}\nAnswer:",
    input_variables=["question", "context"]
)

COT_TEMPLATE = """You are a document expert. Use the context to answer step by step.

Think:
1. What is being asked?
2. What does the context say?
3. What is the accurate answer based only on the context?

Context:
{context}

Question: {question}

Step-by-step answer:"""

RAG_TEMPLATE = """Answer the question using ONLY the context below. Be concise.

Context:
{context}

Question: {question}
Answer:"""

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def load_and_chunk(filepath):
    ext = filepath.rsplit(".", 1)[1].lower()
    if ext == "pdf":
        loader = PyPDFLoader(filepath)
    else:
        loader = TextLoader(filepath, encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return splitter.split_documents(documents)

def build_vectorstore(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    return db

def get_context(query, db, k=5):
    results = db.similarity_search_with_relevance_scores(query, k=k)
    if not results or results[0][1] < 0.1:
        return None, []
    context = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    sources = list(set([os.path.basename(doc.metadata.get("source", "unknown"))
                        for doc, _ in results]))
    return context, sources

def format_json_response(model, raw_answer, sources):
    json_prompt = f"""Return a JSON object with exactly these fields:
- "answer": the answer text
- "confidence": "high", "medium", or "low"
- "sources": {sources}

Answer to format: {raw_answer}

Return ONLY valid JSON, no markdown, no backticks."""
    try:
        raw = model.invoke(json_prompt).content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception:
        return {"answer": raw_answer, "confidence": "medium", "sources": sources}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file. Use PDF, TXT, or MD."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        chunks = load_and_chunk(filepath)
        build_vectorstore(chunks)
        return jsonify({
            "success": True,
            "filename": filename,
            "chunks": len(chunks),
            "message": f"Processed {len(chunks)} chunks from {filename}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    question = data.get("question", "").strip()
    mode     = data.get("mode", "few_shot")  # few_shot | cot | agent

    if not question:
        return jsonify({"error": "No question provided"}), 400
    if not os.path.exists(CHROMA_PATH):
        return jsonify({"error": "No document loaded. Please upload a document first."}), 400

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db    = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    model = ChatOllama(model=MODEL_NAME, temperature=0.1)

    try:
        if mode == "agent":
            # ReAct agent with document + web search tools
            def retrieve(q):
                ctx, srcs = get_context(q, db)
                if not ctx:
                    return "No relevant results found in the document."
                return f"CONTEXT:\n{ctx}\n\nSOURCES: {srcs}"

            retriever_tool = Tool(
                name="DocumentRetriever",
                func=retrieve,
                description="Search the uploaded document. Use this FIRST."
            )
            web_tool = Tool(
                name="WebSearch",
                func=DuckDuckGoSearchRun().run,
                description="Search the web. Use ONLY if document has no answer."
            )
            if not HUB_AVAILABLE:
                return jsonify({"error": "langchainhub not available. Run: pip install langchainhub"}), 500
            react_prompt = hub.pull("hwchase17/react")
            agent = create_react_agent(model, [retriever_tool, web_tool], react_prompt)
            executor = AgentExecutor(
                agent=agent, tools=[retriever_tool, web_tool],
                verbose=False, handle_parsing_errors=True, max_iterations=5
            )
            result = executor.invoke({"input": question})
            return jsonify({
                "answer": result["output"],
                "confidence": "high",
                "sources": ["document + web search"],
                "mode": "ReAct Agent"
            })

        else:
            context, sources = get_context(question, db)
            if not context:
                return jsonify({
                    "answer": "I couldn't find relevant information in the document for this question.",
                    "confidence": "low",
                    "sources": [],
                    "mode": mode
                })

            if mode == "cot":
                prompt_template = ChatPromptTemplate.from_template(COT_TEMPLATE)
                raw = model.invoke(
                    prompt_template.format(context=context, question=question)
                ).content
            else:
                raw = model.invoke(
                    few_shot_prompt.format(question=question, context=context)
                ).content

            result = format_json_response(model, raw, sources)
            result["mode"] = "Chain-of-Thought" if mode == "cot" else "Few-Shot RAG"
            return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
