import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import warnings
from pathlib import Path as p
from pprint import pprint
import pandas as pd
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pinecone
import os

load_dotenv()
warnings.filterwarnings("ignore")
 
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

index_name = "demo01"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name='my_index',
        dimension=1536,
        metric='consine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'
        )
    )

index = pinecone.Index(index_name,host="https://demo01-e2blbbu.svc.aped-4627-b74a.pinecone.io")

model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY,
                               temperature=0.2, convert_system_message_to_human=True)

pdf_loader = PyPDFLoader("resume.pdf")
pages = pdf_loader.load_and_split()
print(len(pages))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
context = "\n\n".join(str(p.page_content) for p in pages)
texts = text_splitter.split_text(context)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

vectors = [embeddings.embed_query(text) for text in texts]

for i, (text, vector) in enumerate(zip(texts, vectors)):
    index.upsert([(str(i), vector)])

vector_index = Pinecone(index, embeddings.embed_query, "text").as_retriever(search_kwargs={"k": 5})

qa_chain = RetrievalQA.from_chain_type(
    model,
    retriever=vector_index,
    return_source_documents=True
)

question = "Can you tell me in which hackathons Satyam has participated?"
result = qa_chain({"query": question})
print(result["result"])
