"""

    File : test_dataframe_query.py

    Author: Tim Schofield
    Date: 13 June 2024

    Examples of DataFrame.query()
    
    Query uses the pandas eval() and is limited in what you can use within it. 
    If you want to use pure SQL you could consider pandasql where the following statement would work for you:
    sqldf("select col_name from df where col_name like 'abc%';", locals())

    Pandas Query Examples: SQL-like queries in dataframes
    https://queirozf.com/entries/pandas-query-examples-sql-like-syntax-queries-in-dataframes


"""
import pandas as pd
from pathlib import Path


# The file to be compared against the authority database
input_folder = "ny_hebarium_location_csv_input"
input_authority_file = "NY_Geopolitical_Lookup_Lists_50000.csv"
input_authority_path = Path(f"{input_folder}/{input_authority_file}")

# irn_eluts    Continent   Country   stateProvince   County
df_authority = pd.read_csv(input_authority_path)
#print(df_authority.head())

# Case sensitive for column names
# Case sensitive for content
result = df_authority[df_authority["Country"]=="Ethiopia"]
# print(result) # [66 rows x 5 columns]

# It repuires the sinle quotes
# This is faster
result = df_authority.query("Country == 'Ethiopia'")
# print(result) [66 rows x 5 columns]

# Using Python variables
this_country = "Ethiopia"
result = df_authority.query("Country == @this_country")
# print(result) # [66 rows x 5 columns]

# Logical operators "or" and "and"
result = df_authority.query("(Country == 'Ethiopia') and (stateProvince == 'Tigray')")
# print(result) # 4 results

# Value in array - the "in" operator
these_countries = ["Ethiopia", "Gabon"]
result = df_authority.query("Country in @these_countries")
# print(result) # [116 rows x 5 columns]

# Value in array - the "not in" operator
these_countries = ["Ethiopia", "Gabon"]
result = df_authority.query("Country not in @these_countries")
# print(result) # [50454 rows x 5 columns]

# Escape column names - no example avaliable
# but you can surround column names with spaces with backticks like ` - TLH corner of the keyboard
# e.g. df.query('`country of birth` == "UK"')

# Is null - note the use of engine="python"
# No real example - but I'll delete an entry to simulate
result = df_authority.query("stateProvince.isnull()", engine="python")
"""
print(result)
   irn_eluts Continent   Country stateProvince                County
2     373035    Africa  Ethiopia           NaN  Mirab Harerge (Zone)
"""

# Is not null - well, no real example
# Problems with using this with a logical conjunction!
result = df_authority.query("stateProvince.notnull()", engine="python")
# print(result)

# "like" is not supported as a keyword so we use col.str.contains()"pattern")
result = df_authority.query("County.str.contains('Bikita Distr.')")
"""
print(result)
    irn_eluts Continent   Country stateProvince         County
10     372354    Africa  Zimbabwe      Masvingo  Bikita Distr.
"""

result = df_authority.query("County.str.contains('Bikita')")
"""
print(result)
    irn_eluts Continent   Country stateProvince         County
10     372354    Africa  Zimbabwe      Masvingo  Bikita Distr.
"""

result = df_authority.query("County.str.contains('ikita')")
print(result)
"""
       irn_eluts Continent   Country stateProvince         County
10        372354    Africa  Zimbabwe      Masvingo  Bikita Distr.
19427     217493      Asia     Japan        Aomori   Kamikita-gun
19817     218022      Asia     Japan      Kumamoto   Ashikita-gun
"""