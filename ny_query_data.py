"""
File : ny_query_data.py

Author: Tim Schofield
Date: 10 June 2024

    # Tests: These are from the authority file, not the transcribed locations, so should work
    # Exact
    # Response from ChatOpenAI: 375150 Africa Ethiopia Oromia Mirab Shewa (Zone) <<<< correct
    continent = "Africa"
    country = "Ethiopia"
    state_province = "Oromia"
    county = "Mirab Shewa (Zone)" 
    
    # A little missing
    # Response from ChatOpenAI: 375150 Africa Ethiopia Oromia Mirab Shewa (Zone) <<<< correct
    continent = "Africa"
    country = ""
    state_province = "Oromia"
    county = "Mirab Shewa"

    # Exact
    # Response from ChatOpenAI: 1036733 Asia China Fujian Putian (City) <<<< correct
    continent = "Asia"
    country = "China"
    state_province = "Fujian"
    county = "Putian (City)"
    
    # A little missing
    # Response from ChatOpenAI: 1036733 Asia China Fujian Putian (City) <<<< correct
    continent = "Asia"
    country = "China"
    state_province = ""
    county = "Putian"

"""
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import openai
import os
from dotenv import load_dotenv

CHROMA_PATH = "chroma" 


load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

def main():
    
    # Prepare the DB - this must have been already been created by running ny_create_database.py
    # Uses text-embedding-ada-002
    # This must be exactly the same as the embedding function used to create the vector database
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB text relevant to "How does Alice meet the Mad Hatter?"
    # By default this is L2 distance
    
    # A little MORE missing
    # Response from ChatOpenAI: 1036733 Asia China Fujian Putian (City) <<<< correct
    continent = ""
    country = "China"
    state_province = ""
    county = "Putian"
    
    
    # The {context} and {question} are filled in by ChatPromptTemplate and its format() method
    prompt_template_for_gpt = """
        Answer the question based only on the following context:
        {context}
        ---
        Answer the question based on the above context: {question}
    """
    
    # For Chroma database
    prompt_for_chroma = (
        f'Find the nearest match to Continent={continent}, Country={country}, State/Province={state_province}, County={county}\n'
        f'Return the matching line together with the irn_eluts number as JSON of structure {{"irn_eluts":"value1", "continent":"value2", "country":"value3", "state_province":"value4", "county":"value5"}}'
    )
    
    # Ask Chroma to find similar chunks
    number_of_answers = 3
    chroma_results = db.similarity_search_with_relevance_scores(prompt_for_chroma, k=number_of_answers)
    # [(doc1, score1), (doc2, score2), (doc3, score3)]

    # These are sorted by score - closest similarity is at the top
    certainty_threshold = 0.5
    if len(chroma_results) == 0 or chroma_results[0][1] < certainty_threshold:
        print(f"Unable to find matching results.")
        return

    # Get the blocks of relevant text back from Chroma and joins them together seperated by newlines and ---
    context_text_from_chroma = "\n\n---\n\n".join([doc.page_content for doc, _score in chroma_results])

    prompt_template = ChatPromptTemplate.from_template(prompt_template_for_gpt)
    
    # This creates a prompt for
    prompt_for_gpt_with_context = prompt_template.format(context=context_text_from_chroma, question=prompt_for_chroma)
    print("#################################################")
    print(prompt_for_gpt_with_context)
    print("#################################################")
    
    # OpenAI takes the blocks of context text returned from the Chroma database
    # And uses them to answer the question
    model = ChatOpenAI()
    response_text = model.predict(prompt_for_gpt_with_context)

    sources = []
    for doc, _score in chroma_results:
        sources.append(doc.metadata)

    formatted_response = f"Response from ChatOpenAI: {response_text}\nSources from Chroma: {sources}"
    
    print("#################################################")
    print(formatted_response)
    print("#################################################")
    
    





if __name__ == "__main__":
    main()
























