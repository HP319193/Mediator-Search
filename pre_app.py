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
from langchain.chains.summarize import load_summarize_chain

from openai import OpenAI
from dotenv import load_dotenv
import os, random, json
from bs4 import BeautifulSoup

from extract import extract_list

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_index = os.getenv("INDEX")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

metadata_list = ['fullname', 'mediator profile on mediate.com', 'mediator Biography', 'mediator state']
metadata_value = ['Name', "Profile", "Biography", "State"]

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
openai_client = OpenAI(api_key=openai_api_key)

practice_list = ', '.join(extract_list())

description = "Extract mediator's practice field that user want to search. Available mediator practice fields are " + practice_list

print(description)
def getMetadata(message):
    tools = [
        {
            "type": "function", 
            "function": {
                "name": "get_info",
                "description": "Extract the information of mediator.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mediator country": {
                            "type": "string",
                            "description": "Extract mediator's country that user want to search."
                            },
                        "mediator city": {
                            "type": "string",
                            "description": "Extract mediator's city that user want to search."
                            },
                        "mediator state": {
                            "type": "string",
                            "description": "Extract mediator's state that user want to search. If both mediator city and mediator state are possible, please extract as mediator state."
                            },
                        "mediator areas of practice": {
                            "type": "array",
                            "description": description,
                            "enum": extract_list()
                            },    
                        }
                    },
                }
            }
    ]

    response = openai_client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
                {"role": "system", "content": f"You are a professional mediation field searcher. Your role is to extract information about mediator from user's message."},
                {"role": "user", "content": message}
            ],
        tools=tools
    )

    print("Message =>", response.choices[0])

    try:
        data = response.choices[0].message.tool_calls[0].function.arguments
        search_status = True
    
    except:
        search_status = False
        data = response.choices[0].message.content

    return {"search_status": search_status, "data": data}

def search(message, history):
    data = getMetadata(message=message)
    search_status = data['search_status']
    answer = ""
    if search_status == True:
        metadata = json.loads(data['data'])
        print(metadata)
        try:
            if 'mediator areas of practice' in metadata:
                practice_data = metadata['mediator areas of practice']
                del metadata['mediator areas of practice']
            elif 'mediator practice field' in metadata:
                practice_data = metadata['mediator practice field']
                del metadata['mediator practice field']
            
        except:
            practice_data = ""

        if "mediator city" in metadata:
            try:
                del metadata['mediator state']
                del metadata['mediator country']
            except:
                pass
        print("metadata =>", metadata)        
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
                                    "description": "The number of mediators that user want to search. If user ask a list of mediators, it means user want to search 3 mediators. If user's message don't have information about the number of mediators, you have to respond with 1.",
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
        try:
            number_str = response.choices[0].message.tool_calls[0].function.arguments
            mediator_num = json.loads(number_str)['mediator']
        except:
            mediator_num = 1
        print(mediator_num)

        template = """"""
        # prompt = "You are a professional mediator information analyzer. You have to write the reason why following mediators are matched to human's message. You shouldn't write mediator's information again. You should't write the mediators in context are the excellent choice or ideal candidate. You have to analyze the mediators at once.  Please respond with no more than 300 characters. "
        prompt = "You are a professional mediator information analyze. You have to analyze the follwing mediators based on human's message. You shouldn't write mediator's information again. You should't write the mediators in context are the excellent choice or ideal candidate. You have to analyze the mediators at once.  Please respond with no more than 300 characters. "
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

        # print(message)
        start_time = time.time()

        pc = Pinecone(api_key=pinecone_api_key)

        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        
        index = pc.Index(pinecone_index)

        results = index.query(
            vector=embeddings.embed_query(message),
            top_k=748,
            filter=metadata,
            include_metadata=True
        )

        print("num of result =>", len(results['matches']))
        end_time = time.time()

        print("Search Time =>", end_time-start_time)
        
        new_docs = []
        new_data = []
        for result in results['matches']:
            data = {}
            for metadata in metadata_list:      
                data[metadata] = result['metadata'][metadata]
            
            if practice_data in result['metadata']['mediator areas of practice']:
                new_data.append(data)

        print(len(new_data))
        random.shuffle(new_data)

        if len(new_data) != 0:
            if practice_data != "" and mediator_num == 1:
                answer += f"I have located a mediator who specializes in {practice_data}.  Here are their details:\n\n"
            elif practice_data != "" and mediator_num > 1:
                answer += f"I have located mediators who specialize in {practice_data}.  Here are their details:\n\n"
            elif practice_data == "" and mediator_num == 1:
                answer += f"I have located a mediator.  Here are their details:\n\n"
            elif practice_data == "" and mediator_num > 1:
                answer += f"I have located mediators.  Here are their details:\n\n"

            for index, new_datum in enumerate(new_data):
                if index < mediator_num:
                    # answer += f"{index+1}\n"
                    content = ""
                    for metadata_index, metadata in enumerate(metadata_list):
                        content += f"<b>{metadata_value[metadata_index]}</b>: {new_datum[metadata]} \n"
                        answer += f"<b>{metadata_value[metadata_index]}</b>: {new_datum[metadata]} \n"

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
        else:
            answer += "I wasn't able to find a mediator that matched your request."
    else:
        answer = data['data']

    return answer

chatbot = gr.Chatbot(avatar_images=["user.png", "bot.jpg"], height=600)

demo = gr.ChatInterface(fn=search, title="Mediate.com Chatbot Prototype", multimodal=False, retry_btn=None, clear_btn=None, undo_btn=None, chatbot=chatbot)

if __name__ == "__main__":
    demo.launch(debug=True)
