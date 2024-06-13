"""

    File : ny_dataframe_query.py

    Author: Tim Schofield
    Date: 12 June 2024

    See test_dataframe_query.py for examples
    https://queirozf.com/entries/pandas-query-examples-sql-like-syntax-queries-in-dataframes


"""
import pandas as pd
import os
from dotenv import load_dotenv
from helper_functions_langchain_rag import get_file_timestamp, is_json, cleanup_json, create_and_save_dataframe
from pathlib import Path
from openai import OpenAI
import re

MODEL = "gpt-4o" # Context window of 128k max_tokens 4096

project_name = "ny_geo_authority_DATAFRAME"

# The file to be compared against the authority database
input_folder = "ny_hebarium_location_csv_input"


input_authority_file = "NY_Geopolitical_Lookup_Lists_50000.csv"
input_authority_path = Path(f"{input_folder}/{input_authority_file}")

input_transcibed_file = "NY_specimens_transcribed.csv" # ###### Note: this is the one that they gave us ######
input_transcibed_path = Path(f"{input_folder}/{input_transcibed_file}")

output_folder = "ny_hebarium_location_csv_output"

batch_size = 20 # saves every
time_stamp = get_file_timestamp()
return_key_list = [ "irn_eluts", "continent", "country", "state_province", "county", "irn", "error", "error_output"]
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

# irn_eluts    Continent   Country   stateProvince   County
df_authority = pd.read_csv(input_authority_path)
# print(df_authority.head())

# irn  DarGlobalUniqueIdentifier DarInstitutionCode  DarCatalogNumber  DarRelatedInformation ... SpeOtherSpecimenNumbers_tab   
df_transcribed = pd.read_csv(input_transcibed_path)
# print(df_transcribed.head())

output_list = []
count = 0
print("####################################### START OUTPUT ######################################")
for index, row in df_transcribed.iterrows():
    
    #row = df.iloc[0]
    
    count+=1
    print(f"\n{count=}")
    
    irn = str(row["irn"])
    continent = str(row["DarContinent"])
    country = str(row["DarCountry"])
    state_province = str(row["DarStateProvince"])
    county = str(row["DarCounty"])
    locality_info = str(row["DarLocality"])

    print(f"{irn} {continent} {country} {state_province} {county} ")

    if continent == "nan" : continent = ""
    if country == "nan": country = ""
    if state_province == "nan" : state_province = ""   
    if county == "nan" : county = ""
    if locality_info == "nan" : locality_info = "" 

    print(f"{irn} {continent} {country} {state_province} {county} ")

    # Search Authority for match(s) with Transcribed line
    # Not sure what to do about "like", i.e. partial matches
    query_string = f""
    if continent != "": query_string = f'(Continent == "{continent}")'
    if country != "": query_string = f'{query_string} and (Country == "{country}")'
    if state_province != "": query_string = f'{query_string} and (stateProvince == "{state_province}")'
    if county != "": query_string = f'{query_string} and (County == "{county}")'
    
    # Case of a spare "and" at the front
    open_bracket_index = query_string.find("(")
    query_string = query_string[open_bracket_index:]
    
    print(f"query_string for authority file: ****'{query_string}'****")
    
    authority_matches = df_authority.query(query_string)
    print(f"Number of matches in authority file: {len(authority_matches)}")
    #print("#########################################")
    #print(authority_matches)
    #print("#########################################")
    
    # Handel no matches
      
    # Then basicaly do conflict resolution using ChatGPT!
    # Basic contextual prompt with out any onther information
    prompt_for_gpt = (
        f'Answer the question based on the following context:\n'
        f'{authority_matches}\n'
        f'Answer the question based on the above context: '
        f'Find the nearest match to Continent="{continent}", Country="{country}", State/Province="{state_province}", County="{county}"\n'
        f'Return the matching line together with the irn_eluts number as JSON of structure {{"irn_eluts":"value1", "continent":"value2", "country":"value3", "state_province":"value4", "county":"value5"}}\n'
        f'Do not return newlines in the JSON'
    )

    #print(prompt_for_gpt)

    gpt_responce = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt_for_gpt}])

    gpt_responce_content = gpt_responce.choices[0].message.content
    gpt_responce_content = cleanup_json(gpt_responce_content)

    # print(gpt_responce_content)

    if is_json(gpt_responce_content) and gpt_responce_content != "{}":
        dict_returned = eval(gpt_responce_content) # JSON -> Dict
        dict_returned["irn"] = irn
        dict_returned["error"] = "OK"
        dict_returned["error_output"] = "NA"
    else:
        if(gpt_responce_content == "{}"):
            error = "GPT RETURNED NO ANSWER"
        else:
            error = "INVALID JSON"  
            
        print(f"{error}: {gpt_responce}")
        dict_returned = dict(empty_output_list)
        dict_returned["irn"] = irn
        dict_returned["error"] = error
        dict_returned["error_output"] = gpt_responce
        
    print(f'OUT ****{dict_returned}************************')

    output_list.append(dict_returned) 
    
    if count % batch_size == 0:
        print(f"WRITING BATCH:{count}")
        output_path = f"{output_folder}/{project_name}_{time_stamp}-{count}.csv"
        create_and_save_dataframe(output_list=output_list, key_list_with_logging=[], output_path_name=output_path)
    
    # Just start by making sure something good comes back from Chroma
    # Not too many options - no need for three answers
    # Test with empty - more answers, smaller chunks
    if count > 60 :break
    
    ###### eo for loop
        
print(f"WRITING BATCH:{count}")
output_path = f"{output_folder}/{project_name}_{time_stamp}-{count}.csv"
create_and_save_dataframe(output_list=output_list, key_list_with_logging=[], output_path_name=output_path)  














