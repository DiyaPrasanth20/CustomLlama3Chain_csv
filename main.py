from flask import Flask, request, jsonify
from langchain_community.llms import Ollama
from pymongo import MongoClient
from chromadb import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Embedding and text splitting functions
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


# Chroma vector database setup
folder_path = "db"


app = Flask( __name__)

cached_llm = Ollama(model ="llama3")


@app.route("/ai", methods=["POST"])
def aiPost():
    print("Post /ai called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    response = cached_llm.invoke(query)
    print(response)

    response_answer = {"answer": response}
    return response_answer







@app.route("/mongodbLoad", methods=["POST"])
def mongodbPost():
    print("Post /mongodb called")

    all_docs = []
    for collection_name in collections:
        collection = db[collection_name]
        docs = list(collection.find({}))
        all_docs.extend(docs)
    
    print(f"Total docs len={len(all_docs)}")

    # Convert docs to text
    texts = [str(doc) for doc in all_docs]

    chunks = text_splitter.split_documents(texts)
    print(f"chunks len={len(chunks)}")

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )

    vector_store.persist()

    response = {
        "status": "Successfully Uploaded",
        "total_docs": len(all_docs),
        "chunks": len(chunks),
    }
    return jsonify(response)




                               

def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == "__main__":
    start_app()