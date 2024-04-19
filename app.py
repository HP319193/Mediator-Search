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

from openai import OpenAI
from dotenv import load_dotenv
import os, random, json
from bs4 import BeautifulSoup

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_index = os.getenv("INDEX")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

metadata_list = ['fullname', 'mediator email', 'mediator profile on mediate.com', 'mediator Biography']
metadata_value = ['Name', "Email", "Profile", "Biography"]

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
openai_client = OpenAI(api_key=openai_api_key)
def search(message, history):
    tools = [
            {
                "type": "function", 
                "function": {
                    "name": "mediator_search",
                    "description": "Extract how many mediators user want to search.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "mediator": {
                                "type": "number",
                                "description": "The number of mediators that user want to search",
                                "default": 1
                            }
                        },
                        "required": ["mediator"]
                    }
                }
            }
        ]
    
    response = openai_client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                    {"role": "system", "content": "Please extract how many mediators users want to search."},
                    {"role": "user", "content": message}
                ],
                tools=tools,
    )

    number_str = response.choices[0].message.tool_calls[0].function.arguments
    
    mediator_num = json.loads(number_str)['mediator']

    print(mediator_num)
    template = """"""
    prompt = "You are a professional mediator information analyzer. You have to write why the following context is related to human's message. Please write 3 or 4 sentences."
    
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

    print(message)
    start_time = time.time()

    pc = Pinecone(api_key=pinecone_api_key)

    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    
    index = pc.Index(pinecone_index)

    results = index.query(
        vector=embeddings.embed_query(message),
        top_k=50,
        include_metadata=True
    )
    end_time = time.time()

    print("Search Time =>", end_time-start_time)
    
    new_docs = []
    new_data = []
    for result in results['matches']:
        if result['score'] > 0.75:
            data = {}
            for metadata in metadata_list:                
                data[metadata] = BeautifulSoup(result['metadata'][metadata], "html.parser").get_text()
            new_data.append(data)
    
    print(len(new_data))
    random.shuffle(new_data)

    answer = ""
    for index, new_datum in enumerate(new_data):
        if index < mediator_num:
            answer += f"{index+1}\n"
            content = ""
            for metadata_index, metadata in enumerate(metadata_list):
                content += f"{metadata_value[metadata_index]}: {new_datum[metadata]} \n"
                answer += f"{metadata_value[metadata_index]}: {new_datum[metadata]} \n"

            answer += "\n\n"
            new_doc = Document(page_content=answer)
            new_docs.append(new_doc)
        else:
            break

    chat_openai = ChatOpenAI(model='gpt-4-1106-preview', 
            openai_api_key=openai_api_key)

    print(new_docs)
    chain = load_qa_chain(chat_openai, chain_type="stuff",  prompt=prompt, memory=memory)
    start_time = time.time()
    output = chain({"input_documents": new_docs, "human_input": message}, return_only_outputs=False)
    end_time = time.time()

    print("Query Time =>", end_time-start_time)
    
    answer += f"Why appropriate: {output['output_text']}"
    return answer

chatbot = gr.Chatbot(avatar_images=["user.png", "bot.jpg"], height=600)

demo = gr.ChatInterface(fn=search, title="Mediate.com Chatbot Prototype", multimodal=False, retry_btn=None, clear_btn=None, undo_btn=None, chatbot=chatbot)

if __name__ == "__main__":
    demo.launch(debug=True)