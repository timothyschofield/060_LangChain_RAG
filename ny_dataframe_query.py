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


load_dotenv()
try:
    my_api_key = os.environ['OPENAI_API_KEY']          
    client = OpenAI(api_key=my_api_key)
except Exception as ex:
    print("Exception:", ex)
    exit()

# irn_eluts    Continent   Country   stateProvince   County
df_authority = pd.read_csv(input_authority_path)
#print(df_authority.head()

# irn  DarGlobalUniqueIdentifier DarInstitutionCode  DarCatalogNumber  DarRelatedInformation ... SpeOtherSpecimenNumbers_tab   
df_transcribed = pd.read_csv(input_transcibed_path)
# print(df_transcribed.head())

output_list = []
count = 0
print("####################################### START OUTPUT ######################################")
for index, row in df_transcribed.iterrows():
    
    #row = df.iloc[0]
    
    count+=1
    print(f"{count=}")
    
    irn = row["irn"]  
    continent = row["DarContinent"]
    country = row["DarCountry"]
    state_province =  row["DarStateProvince"]
    county = row["DarCounty"]

    # Get a line from Transcribed and get match(s) in Authority

    # Fake line from Transcribed
    irn = "999999"
    continent = "Africa"
    country = "Zimbabwe"
    state_province =  "Midlands Province"
    county = ""  # Shurugwi Distr.

    # Search Authority for match(s) with Transcribed line
    # Not sure what to do about "like", i.e. partial matches
    query_string = f""
    if continent != "": query_string = f"(Continent == '{continent}')"
    if country != "": query_string = f"{query_string} and (Country == '{country}')"
    if state_province != "": query_string = f"{query_string} and (stateProvince == '{state_province}')"
    if county != "": query_string = f"{query_string} and (County == '{county}')"
    
    # Case of a spare "and" at the front
    open_bracket_index = query_string.find("(")
    query_string = query_string[open_bracket_index:]
    
    print(f"****'{query_string}'****")
    
    authority_matches = df_authority.query(query_string)
    print(authority_matches)
    
    
    # Then basicaly do conflict resolution using ChatGPT!

    exit()














