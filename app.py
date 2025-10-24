from flask import Flask, render_template, request, Response
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
from src.prompt import *   # make sure system_prompt includes {context}
import os, json

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY or ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY or ""

embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ----- Non-streaming chain (kept for /get) -----
chatModel = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ----- Streaming chain (for /stream) -----
# Format docs into a single context string
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

stream_llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

stream_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "input": RunnablePassthrough(),
    }
    | prompt
    | stream_llm
    | StrOutputParser()   # ensures text chunks
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    response = rag_chain.invoke({"input": msg})
    return str(response["answer"])

# NEW: SSE streaming endpoint
@app.route("/stream", methods=["GET"])
def stream():
    q = request.args.get("msg", "").strip()
    if not q:
        return Response("Missing msg", status=400)

    def gen():
        # optional retry hint for EventSource
        yield "retry: 1000\n\n"
        for token in stream_chain.stream(q):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: {\"done\": true}\n\n"

    return Response(gen(), mimetype="text/event-stream")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
