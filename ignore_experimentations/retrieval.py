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


loader= TextLoader("transcript.txt")
documents= loader.load()
text_splitter= RecursiveCharacterTextSplitter(
    chunk_size= 500,
    chunk_overlap=0
    )

docs= text_splitter.split_documents(documents)
 
embedding= MistralAIEmbeddings(api_key="nGk7g7h64VrSy11XHypNMvyFtDJ4ziRN")
embedding.model="mistral-embed"
embeddings_list = embedding.embed_documents(docs)
library= FAISS.from_documents(docs, embeddings_list)

user_query= input("Shoot your question")
query_answer= library.similarity_search(user_query)

retriever= library.as_retriever()
qa= retrieval_qa.from_chain_type(llm= OpenAI(), chain_type= "stuff", retriever= retriever)
retriever_query= user_query
results= qa.invoke(retriever_query)
print(results)
