"""
File : ny_query_data.py

Author: Tim Schofield
Date: 10 June 2024

    # Tests: These are from the authority file, not the transcribed locations, so they should work
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

   # A little MORE missing
    # Response from ChatOpenAI: 1036733 Asia China Fujian Putian (City) <<<< correct
    continent = ""
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

from openai import OpenAI
from pathlib import Path 
from helper_functions_langchain_rag import get_file_timestamp, is_json, cleanup_json, create_and_save_dataframe
import pandas as pd
from math import isnan

CHROMA_PATH = "chroma"
MODEL = "gpt-4o" # Context window of 128k max_tokens 4096

project_name = "ny_geo_authority"

# The file to be compared against the authority database
input_folder = "ny_hebarium_location_csv_input"
input_file = "NY_specimens_transcribed.csv"         # Note: this is the one that they gave us
input_path = Path(f"{input_folder}/{input_file}")

output_folder = "ny_hebarium_location_csv_output"
batch_size = 5 # saves every
time_stamp = get_file_timestamp()

return_key_list = ["irn", "error", "irn_eluts", "continent", "country", "state_province", "county", "error_output"]
empty_output_list = dict()
for key in return_key_list:
    empty_output_list[key] = "none"


load_dotenv()
try:
    my_api_key = os.environ['OPENAI_API_KEY']          
    client = OpenAI(api_key=my_api_key)
except Exception as ex:
    print("Exception:", ex)
    exit()

# Prepare the DB - this must have been already been created by running ny_create_database.py
# Uses text-embedding-ada-002
# This must be exactly the same as the embedding function used to create the vector database
embedding_function = OpenAIEmbeddings()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# The {context} and {question} are filled in by ChatPromptTemplate and its format() method
prompt_template_for_gpt = """
    Answer the question based only on the following context:
    {context}
    ---
    Answer the question based on the above context: {question}
"""

df = pd.read_csv(input_path)
output_list = []
count = 0
for index, row in df.iterrows():
    count+=1
    
    irn = row["irn"]  
    continent = row["DarContinent"]
    country = row["DarCountry"]
    state_province =  row["DarStateProvince"]
    county = row["DarCounty"]

    # Get rid if nan coming in from spreadsheet
    # Strange fact: nan != nan is True
    if continent != continent : continent = ""
    if country != country : country = ""
    if state_province != state_province : state_province = ""   
    if county != county : county = ""   
    
    print(f"IN ****{continent}, {country}, {state_province}, {county}****")

    # For Chroma database
    prompt_for_chroma = (
        f'Find the nearest match to Continent={continent}, Country={country}, State/Province={state_province}, County={county}\n'
        f'Return the matching line together with the irn_eluts number as JSON of structure {{"irn_eluts":"value1", "continent":"value2", "country":"value3", "state_province":"value4", "county":"value5"}}'
        f'Do not return newlines in the JSON'
    )
    
    # Search Chroma to: "Find the nearest match to Continent=Asia, Country=China,..."
    # By default this uses L2 distance
    # Return similar chunks
    number_of_answers = 3
    chroma_results = db.similarity_search_with_relevance_scores(prompt_for_chroma, k=number_of_answers)
    # [(doc1, score1), (doc2, score2), (doc3, score3)]

    # These are sorted by score - closest similarity is at the top
    certainty_threshold = 0.5
    if len(chroma_results) == 0 or chroma_results[0][1] < certainty_threshold:
        print(f"###################### Unable to find matching results. ######################")
        # Have to deal with this
    else:
        pass
    
    # Get the chunks of relevant text back from Chroma and joins them together seperated by newlines and ---
    context_text_from_chroma = "\n\n---\n\n".join([doc.page_content for doc, _score in chroma_results])

    prompt_template = ChatPromptTemplate.from_template(prompt_template_for_gpt)
    
    # This creates a prompt
    prompt_for_gpt_with_context = prompt_template.format(context=context_text_from_chroma, question=prompt_for_chroma)
    # print(f"{prompt_for_gpt_with_context=}")
    
    # OpenAI takes the blocks of context text returned from the Chroma database
    # And uses them to answer the question
    gpt_responce = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt_for_gpt_with_context}])
    # ChatCompletion object returned - how to handle errors?

    gpt_responce_content = gpt_responce.choices[0].message.content
    gpt_responce_content = cleanup_json(gpt_responce_content)
    
    if is_json(gpt_responce_content):
        dict_returned = eval(gpt_responce_content) # JSON -> Dict
        dict_returned["irn"] = irn
        dict_returned["error"] = "OK"
        dict_returned["error_output"] = "none"
    else:
        print(f"INVALID JSON: {gpt_responce}")  
        dict_returned = dict(empty_output_list)
        dict_returned["irn"] = irn
        dict_returned["error"] = "INVALID JSON"
        dict_returned["error_output"] = gpt_responce
    
    print(f'OUT ****{dict_returned}****')
        
    output_list.append(dict_returned) 
    
    if count % batch_size == 0:
        print(f"WRITING BATCH:{count}")
        output_path = f"{output_folder}/{project_name}_{time_stamp}-{count}.csv"
        create_and_save_dataframe(output_list=output_list, key_list_with_logging=[], output_path_name=output_path)
    
    if count > 15 :break
    
    ###### eo for loop
        
print(f"WRITING BATCH:{count}")
output_path = f"{output_folder}/{project_name}_{time_stamp}-{count}.csv"
create_and_save_dataframe(output_list=output_list, key_list_with_logging=[], output_path_name=output_path)     
        
print("####################################### END OUTPUT ######################################")

























