"""
File : create_datbase.py

Author: Tim Schofield
Date: 06 June 2024

Run this program first to create the vector embedding of the data in "data/books/*.md"

"""
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil


load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    """
    So she was considering in her own mind (as well as she could, for the
    hot day made her feel very sleepy and stupid), whether the pleasure of
    making a daisy-chain would be worth the trouble of getting up and
    picking the daisies, when suddenly a White Rabbit with pink eyes ran
    close by her.
    
    289 characters
    """
    
    print(document.metadata) # {'source': 'data/books/alice_in_wonderland.md', 'start_index': 1654}

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH)
    # db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
    # Saved 801 chunks to chroma.

if __name__ == "__main__":
    main()


