import csv

from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

def summarize(text):
    prompt_template = """Write a concise summary of the following context. Summary should be up to 350 characters.
    Context: "{text}"
    CONCISE SUMMARY:"""

    prompt = PromptTemplate.from_template(prompt_template)

    llm = ChatOpenAI(temperature=0, model_name="gpt-4-1106-preview", api_key=openai_api_key)
    chain = load_summarize_chain(llm, chain_type="stuff")

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
    return stuff_chain.run([Document(page_content=text)])

# Specify the input CSV file
csvfile = 'updated.csv'

# Read the data from the input CSV file
with open(csvfile, 'r') as file:
    csv_reader = csv.DictReader(file)
    rows = list(csv_reader)

# Modify the data as needed
for row in rows:
    row['mediator Biography'] = summarize(row['mediator Biography'])

with open(csvfile, 'w', newline='') as file:
    fieldnames = csv_reader.fieldnames
    csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
    csv_writer.writeheader()
    csv_writer.writerows(rows)

print("CSV file content has been successfully changed.")