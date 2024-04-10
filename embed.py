import streamlit as st
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders.base import BaseLoader
from langchain.docstore.document import Document
from langchain_pinecone import PineconeVectorStore

from typing import Dict, List, Optional
from dotenv import load_dotenv
import os, csv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_index = os.getenv("INDEX")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

class MetaDataCSVLoader(BaseLoader):
    def __init__(
        self,
        file_path: str,
        source_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,   
        content_columns: Optional[List[str]] =None ,  
        csv_args: Optional[Dict] = None,
        encoding: Optional[str] = None,
    ):
        self.file_path = file_path
        self.source_column = source_column
        self.encoding = encoding
        self.csv_args = csv_args or {}
        self.content_columns= content_columns
        self.metadata_columns = metadata_columns

    def load(self) -> List[Document]:
        docs = []
        with open(self.file_path, newline="", encoding=self.encoding) as csvfile:
            csv_reader = csv.DictReader(csvfile, **self.csv_args)  # type: ignore
            for i, row in enumerate(csv_reader):
                if self.content_columns: 
                    content = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in row.items() if k in self.content_columns)
                else: 
                    content = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in row.items())
                try:
                    source = (
                        row[self.source_column]
                        if self.source_column is not None
                        else self.file_path
                    )
                except KeyError:
                    raise ValueError(
                        f"Source column '{self.source_column}' not found in CSV file."
                    )
                metadata = {"source": source, "row": i}
                # ADDED TO SAVE METADATA
                if self.metadata_columns:
                    for k, v in row.items():
                        if k in self.metadata_columns:
                            metadata[k] = v
                # END OF ADDED CODE
                doc = Document(page_content=content, metadata=metadata)
                docs.append(doc)

        return docs


csv_file_uploaded = st.file_uploader(label="Upload your CSV File here")

if csv_file_uploaded is not None:
    def save_file_to_folder(uploadedFile):
        save_folder = 'content'
        save_path = Path(save_folder, uploadedFile.name)
        with open(save_path, mode='wb') as w:
            w.write(uploadedFile.getvalue())

        if save_path.exists():
            st.success(f'File {uploadedFile.name} is successfully saved!')

            with open(os.path.join('content/', csv_file_uploaded.name), 'r') as file:

                csv_reader = csv.reader(file)

                # Read the headers from the CSV file
                headers = next(csv_reader)

            filtered_headers= list(filter(lambda x: x != '', headers))

            loader = MetaDataCSVLoader(os.path.join('content/', csv_file_uploaded.name), 
                metadata_columns=filtered_headers, encoding = "utf-8")
            data = loader.load()

            # Pinecone.from_documents(data, embeddings, index_name=index_name)
            PineconeVectorStore.from_documents(data, embeddings, index_name=pinecone_index)

    save_file_to_folder(csv_file_uploaded)