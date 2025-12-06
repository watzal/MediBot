import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain_pinecone import PineconeEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pinecone import Pinecone
from src.prompt import *
from src.helper import download_embedding
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

st.title("Medical Chatbot")

PINECONE_API_KEY=st.secrets['PINECONE_API_KEY']
GEMINI_API_KEY=st.secrets['GEMINI_API_KEY']

# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

if not PINECONE_API_KEY:
    st.error("Missing PINECONE_API_KEY in Streamlit Secrets")
if not GEMINI_API_KEY:
    st.error("Missing GEMINI_API_KEY in Streamlit Secrets")
    
embeddings = download_embedding()


pc = Pinecone(api_key = PINECONE_API_KEY)
index_name = "medical-chatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
    
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":5})

model = ChatGoogleGenerativeAI(model='gemini-2.5-pro',api_key=GEMINI_API_KEY)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


with st.form("my_form"):
    msg = st.text_area(
        "Enter text:"
    )
    submitted = st.form_submit_button("Submit")
  
    if submitted:
        response = rag_chain.invoke({"input":msg})
        st.info(response['answer'])

