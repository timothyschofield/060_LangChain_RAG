"""
File : ny_create_datebase.py

Author: Tim Schofield
Date: 10 June 2024

https://dev.to/eteimz/understanding-langchains-recursivecharactertextsplitter-2846



The RecursiveCharacterTextSplitter takes a large text and splits it based on a specified chunk size. 
It does this by using a set of characters. 
The default characters provided to it are ["\n\n", "\n", " ", ""].

It takes in the large text then tries to split it by the first character \n\n. 
If the first split by \n\n is still large then it moves to the next character which is \n and tries to split by it. 
If it is still larger than our specified chunk size it moves to the next character 
in the set until we get a split that is less than our specified chunk size.


"""

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil


load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = "chroma"
DATA_PATH = "data/csvs"


def main():
    generate_data_store()
    
def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.csv")
    documents = loader.load()
    return documents

"""
    Split 1 documents into 10258 chunks - full 50000 database.
    1042204
    Asia
    China
    Guangdong
    Shantou (City)


    1037438
    Asia
    China
    Guangdong
    Yangjiang (City)
    ...

    'source': 'data/csvs/NY_Geopolitical_Lookup_Lists_1000.csv', 'start_index': 2747}
    
"""
def split_text(documents: list[Document]):
    
    # The characters we use for splitting will be ['\n\n', '\n', ' ', '']
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0, length_function=len, add_start_index=True)

    chunks = text_splitter.split_documents(documents)
    
    this_chunk = chunks[10]
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    print(this_chunk.page_content)
    print(this_chunk.metadata)  # {'source': 'data/csvs/NY_Geopolitical_Lookup_Lists_1000.csv', 'start_index': 2747}

    return chunks
   
def save_to_chroma(chunks: list[Document]):
    
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    # This must be exactly the same as the embedding function used to query the database
    # Uses text-embedding-ada-002
    db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH)
    
    # Saved 193 chunks to chroma.
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()





















