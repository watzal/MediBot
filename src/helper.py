from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from typing import List
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# os.chdir('../')

def get_file(data):
    loader = DirectoryLoader(data,glob='*.pdf',loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def filter_data(docs: List[Document]) -> List[Document]:
    filtered_data: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        filtered_data.append(
            Document(page_content=doc.page_content, metadata = {"source":src})
        )
    return filtered_data

def chunk_docs(filtered_data: List[Document]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    chunked_docs = splitter.split_documents(filtered_data)
    return chunked_docs

def download_embedding():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding = HuggingFaceEmbeddings(model_name=model_name,model_kwargs={"device": "cpu"},encode_kwargs={"normalize_embeddings": False})
    return embedding