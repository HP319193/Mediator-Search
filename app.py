import gradio as gr
import os
import time

from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

from dotenv import load_dotenv
import os, random

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_index = os.getenv("INDEX")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

metadata_list = ['fullname', 'mediator email', 'mediator profile on mediate.com', 'mediator Biography']

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

def search(message, history):
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

    print(message)
    start_time = time.time()
    docs = docsearch.similarity_search(message, k=15)
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
    output = chain({"input_documents": new_docs, "human_input": message}, return_only_outputs=False)
    end_time = time.time()

    print("Query Time =>", end_time-start_time)
    print(output)
    return output["output_text"]

chatbot = gr.Chatbot(avatar_images=["user.png", "bot.jpg"], height=600)

demo = gr.ChatInterface(fn=search, title="Mediate.com Chatbot Prototype", multimodal=False, retry_btn=None, clear_btn=None, undo_btn=None, chatbot=chatbot)

if __name__ == "__main__":
    demo.launch(debug=True)