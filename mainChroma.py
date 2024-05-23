import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

import warnings
from pathlib import Path as p
from pprint import pprint
import pandas as pd
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings

warnings.filterwarnings("ignore")


GOOGLE_API_KEY="AIzaSyCoIdSzzg5-YdDA2tWC-cG0wYbuJiTKcMI"
model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=GOOGLE_API_KEY,
                             temperature=0.2,convert_system_message_to_human=True)


pdf_loader = PyPDFLoader("syllabus.pdf")
pages = pdf_loader.load_and_split()
# print(pages[3].page_content)

print(len(pages))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
context = "\n\n".join(str(p.page_content) for p in pages)
texts = text_splitter.split_text(context)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)

vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":5})

qa_chain = RetrievalQA.from_chain_type(
    model,
    retriever=vector_index,
    return_source_documents=True
)

question = "Can you tell me the name of chapters of the subject Automata?"
result = qa_chain({"query": question})
print(result["result"])
