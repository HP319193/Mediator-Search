import streamlit as st
from pathlib import Path
from streamlit_chat import message
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os
import time
from typing import Dict, List, Optional
from langchain.document_loaders.base import BaseLoader
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

from dotenv import load_dotenv
import os, random

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_index = os.getenv("INDEX")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

metadata_list = ['fullname', 'mediator email', 'mediator profile on mediate.com', 'mediator Biography']
st.title('Mediate.com Chatbot Prototype')

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def generate_response(user_query):
    template = """"""
    prompt = "You are a professional mediator search expert. Your role is to generate answer to question depending on the context. Your answer must be from context. You answer must include mediator's email, mediator profile on mediate.com, mediator Biography and the reason why this mediator is appropriate to human input. You have to mention that current search result is not completed and chatbot can search different mediators or mediator."
    
    end = """Context: {context}
        Chat history: {chat_history}
        Human: {human_input}
        Your Response as Chatbot:"""
    template += prompt + end

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input", "context"], 
        template=template
        )

    memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
    # docsearch = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    docsearch = PineconeVectorStore.from_existing_index(index_name = pinecone_index, embedding = embeddings)

    print(user_query)
    start_time = time.time()
    docs = docsearch.similarity_search(user_query, k=15)
    end_time = time.time()

    print("Search Time =>", end_time-start_time)
    
    new_docs = []
    for doc in docs:
        page_content = ""
        for metadata in metadata_list:
            page_content += f"{metadata}: {doc.metadata[metadata]}" + "\n"

        new_doc =  Document(page_content=page_content)

        new_docs.append(new_doc)

    random.shuffle(new_docs)
    chat_openai = ChatOpenAI(model='gpt-4-1106-preview', 
            openai_api_key=openai_api_key)

    print(new_docs)
    chain = load_qa_chain(chat_openai, chain_type="stuff",  prompt=prompt, memory=memory)
    start_time = time.time()
    output = chain({"input_documents": new_docs, "human_input": user_query}, return_only_outputs=False)
    end_time = time.time()

    print("Query Time =>", end_time-start_time)
    print(output)
    return output["output_text"]

user_input = st.text_input("You: ","", key="input")

if user_input:
    output = generate_response(user_input) 
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')