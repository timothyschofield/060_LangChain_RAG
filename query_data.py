"""
File : query_data.py

Author: Tim Schofield
Date: 06 June 2024

First run the create_database.py
This creates the vector embedding of data/books/alice_in_wonderland.md

The query_text = "How does Alice meet the Mad Hatter?"
is used twice:

1. By the Chroma database searches for blocks of text similar to query_text
returns with a similarity_search_with_relevance_scores and 
returns a references to the document in which they were found
returns with similarity scores

This creates the context_text

2. ChatGPT then takes the context_text and the query_text to answer the question.

"""

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import openai
import os
from dotenv import load_dotenv

CHROMA_PATH = "chroma"

# The {context} and {question} are filled in by ChatPromptTemplate and its format() method
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']


def main():
    
    # Prepare the DB - this must have been already been created by running create_database.py
    # Uses text-embedding-ada-002
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)


    # Search the DB text relevant to "How does Alice meet the Mad Hatter?"
    # By default this is L2 distance
    query_text = "How does Alice meet the Mad Hatter?"
    number_of_answers = 3
    chroma_results = db.similarity_search_with_relevance_scores(query_text, k=number_of_answers)
    # [(doc1, score1), (doc2, score2), (doc3, score3)]
    
    
    if len(chroma_results) == 0 or chroma_results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    # Get the blocks of relevant text back from Chroma and joins them together seperated by newlines and ---
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in chroma_results])
    
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("#################################################")
    print(prompt)
    print("#################################################")

    """
    'Human: 
    
    Answer the question based only on the following context:
    
    “In that direction,” the Cat said, waving its right paw round, “lives
    a Hatter: and in that direction,” waving the other paw, “lives a
    March Hare. Visit either you like: they’re both mad.”
    
    “But I don’t want to go among mad people,” Alice remarked.
    
    ---
    
    Alice waited a little, half expecting to see it again, but it did not
    appear, and after a minute or two she walked on in the direction in
    which the March Hare was said to live. “I’ve seen hatters before,” she
    said to herself; “the March Hare will be much the most interesting, and
    
    ---
    
    “But I don’t want to go among mad people,” Alice remarked.
    
    “Oh, you can’t help that,” said the Cat: “we’re all mad here. I’m mad.
    You’re mad.”
    
    “How do you know I’m mad?” said Alice.
    
    “You must be,” said the Cat, “or you wouldn’t have come here.”
    
    ---
    
    Answer the question based on the above context: How does Alice meet the Mad Hatter?'
   
    """
  
    # OpenAI takes the blocks of context text returned from the Chroma database
    # And uses them to answer the question
    model = ChatOpenAI()
    response_text = model.predict(prompt)

    sources = []
    for doc, _score in chroma_results:
        sources.append(doc.metadata)

    formatted_response = f"Response from ChatOpenAI: {response_text}\nSources from Chroma: {sources}"
    
    print("#################################################")
    print(formatted_response)
    print("#################################################")

if __name__ == "__main__":
    main()
