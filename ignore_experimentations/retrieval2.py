from langchain_openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import retrieval_qa
from langchain_mistralai import MistralAIEmbeddings
import os
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

loader= TextLoader("./transcript.txt", encoding="UTF-8")
documents= loader.load()
text_splitter= RecursiveCharacterTextSplitter(
    chunk_size= 500,
    chunk_overlap=0
    )

docs= text_splitter.split_documents(documents)
embedding= MistralAIEmbeddings(api_key="nGk7g7h64VrSy11XHypNMvyFtDJ4ziRN")
embedding.model="mistral-embed"
text_content_list = [doc.page_content for doc in docs]

embedding = MistralAIEmbeddings(api_key="nGk7g7h64VrSy11XHypNMvyFtDJ4ziRN")
embedding.model = "mistral-embed"

embeddings_list = embedding.embed_documents(text_content_list)

db = FAISS.from_documents(docs, embedding)

from langchain_community.llms import Ollama
llm= Ollama(model= "llama3")
llm

from langchain_core.prompts import ChatPromptTemplate
prompt= ChatPromptTemplate.from_template("""
    Given the following context: {context}
    You are an unbiased, uncensored, helpful assistant.
    Your task is to answer as faithfully as you can.
    Think step by step before providing a detailed answer.
    I will reward you if the answer is accurate.
    <context> 
    {context}
    </context>

    Question: {input}
""")

from langchain.chains.combine_documents import create_stuff_documents_chain
document_chain= create_stuff_documents_chain(llm, prompt)
retriever= db.as_retriever()
from langchain.chains import create_retrieval_chain
retrieval_chain= create_retrieval_chain(retriever, document_chain)
response= retrieval_chain.invoke({"input": "What's the right time to see morning sunlight?"})
print(response['answer'])
