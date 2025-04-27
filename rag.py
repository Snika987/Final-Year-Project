from fastapi import FastAPI, HTTPException
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_mistralai import ChatMistralAI
from langchain.prompts import PromptTemplate
import os


app = FastAPI()

# Set environment variables for API keys
os.environ['MISTRAL_API_KEY'] = "key-here"
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "key here"

# Load and process documents
def load_docs(directory=r"C:\Users\San_D\Desktop\Final Year Project\Project API\KVASIR & RAG STREAMLIT\DATA"):
    loader = PyPDFDirectoryLoader(directory)
    return loader.load()

documents = load_docs()

def split_docs(documents, chunk_size=512, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

docs = split_docs(documents)

# Initialize embeddings and vector store
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
index = FAISS.from_documents(docs, embeddings)

def get_similar_docs(query, k=4):
    return index.similarity_search(query, k=k)

# Load LLM
llm = ChatMistralAI(model="mistral-large-latest", temperature=0.7, max_tokens=150)

# Define prompt
med_prompt = PromptTemplate.from_template(
    """
    You are an expert medical assistant.
    Based only on the summaries provided below, answer the given question.
    Give all details as much as possible.
    If you don't know the answer, respond with "I don't know".
    
    Source material: {context}
    
    Question: {question}
    Answer:
    """
)

chain = med_prompt | llm

'''@app.post("/query")
def get_answer(query: str):
    relevant_docs = get_similar_docs(query)
    response = chain.invoke({"context": relevant_docs, "question": query})
    return {"answer": response.content}'''


@app.get("/query")
def get_answer(query: str):
    relevant_docs = get_similar_docs(query)
    response = chain.invoke({"context": relevant_docs, "question": query})
    return {"answer": response.content}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
