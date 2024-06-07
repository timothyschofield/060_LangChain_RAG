import argparse
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import openai
import os
from dotenv import load_dotenv

context = "data/books/alice_in_wonderland.md"
question = "How did Alice meet the Mad Hatter?"

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# First run the create_database.py. This creates the vector embedding of data/books/alice_in_wonderland.md
# Then in the terminal type:
# python3 query_data.py "How does Alice meet the Mad Hatter?"

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

print("###################################################")
print(PROMPT_TEMPLATE)
print("###################################################")


def main():
    """
    # Create CLI.
    parser = argparse.ArgumentParser()
    
    parser.add_argument("query_text", type=str, help="The query text.")
    
    args = parser.parse_args()
    query_text = args.query_text
    """
    
    query_text = 'How does Alice meet the Mad Hatter?'

    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
     
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return


    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOpenAI()
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()
