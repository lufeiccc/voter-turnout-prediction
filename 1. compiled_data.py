import pandas as pd
import requests
from bs4 import BeautifulSoup
import re

#EDUCATION, UNEMPLOYMENT, AND INCOME DATA


# Load and clean 2016 education data
edu2016 = pd.read_csv('data/EduA2016.csv')
edu2016_filtered = edu2016[
    (edu2016['Label (Grouping)'] == 'Percent high school graduate or higher') | 
    (edu2016['Label (Grouping)'] == "Percent bachelor's degree or higher")
]
edu2016_filtered = edu2016_filtered.copy()
edu2016_filtered['Label (Grouping)'] = edu2016_filtered['Label (Grouping)'].replace({
    'Percent high school graduate or higher': 'High School',
    "Percent bachelor's degree or higher": "Bachelor's or Higher"
})
percent_columns_2016 = [col for col in edu2016_filtered.columns if '!!Percent!!Estimate' in col]
edu2016_filtered = edu2016_filtered[['Label (Grouping)'] + percent_columns_2016]
edu2016_filtered.columns = ['Label (Grouping)'] + [col.split(',')[0].replace("County", "").strip() for col in percent_columns_2016]
edu2016_pivoted = edu2016_filtered.set_index('Label (Grouping)').T
edu2016_pivoted.reset_index(inplace=True)
edu2016_pivoted.rename(columns={'index': 'County'}, inplace=True)

# Load and clean 2020 education data
edu2020 = pd.read_csv('data/EduA2020.csv')
county_columns = [col for col in edu2020.columns if '!!Percent!!Estimate' in col]


"""
Extract the county name from the column header.
For example: if the column header is "Adams County, Pennsylvania!!Percent!!Estimate",
this function extracts "Adams" from the column header.
"""
def extract_county_name(full_name):
    return full_name.split(',')[0].replace('County', '').strip()


"""
Convert a percentage string to a float (e.g., "40.0%" becomes 40.0).
Returns None if conversion fails (e.g., "N/A").

Used to process education percentages for ages 18-24 and 25+,
enabling calculations of combined education levels across age groups.
"""
def parse_percentage(value):
    try:
        return float(value.strip('%'))
    except ValueError:
        return None

county_data = {'Label (Grouping)': ['High School', "Bachelor's or Higher"]}
for county in county_columns:
    high_school_18_24 = parse_percentage(edu2020.loc[3, county])
    high_school_25_over = parse_percentage(edu2020.loc[9, county])
    bachelor_18_24 = parse_percentage(edu2020.loc[5, county])
    bachelor_25_over = parse_percentage(edu2020.loc[15, county])
    combined_high_school = round(high_school_18_24 + high_school_25_over, 1) if high_school_18_24 and high_school_25_over else None
    combined_bachelor_or_higher = round(bachelor_18_24 + bachelor_25_over, 1) if bachelor_18_24 and bachelor_25_over else None
    county_name = extract_county_name(county)
    county_data[county_name] = [f"{combined_high_school}%" if combined_high_school else None, 
                                f"{combined_bachelor_or_higher}%" if combined_bachelor_or_higher else None]

edu2020_filtered = pd.DataFrame(county_data).set_index('Label (Grouping)').T
edu2020_filtered.reset_index(inplace=True)
edu2020_filtered.rename(columns={'index': 'County'}, inplace=True)

# Combine 2016 and 2020 education data
edu2016_pivoted['Year'] = 2016
edu2020_filtered['Year'] = 2020
edu2016_pivoted.set_index(['County', 'Year'], inplace=True)
edu2020_filtered.set_index(['County', 'Year'], inplace=True)
combined_edu_data = pd.concat([edu2016_pivoted, edu2020_filtered])
combined_edu_data.sort_index(level=['County', 'Year'], inplace=True)



# Load and extract 2016 unemployment data
unem2016 = pd.read_csv('data/unemIncome2016.csv')

"""
Extracts unemployed data for counties from the specified row of a DataFrame.
Parameters:
- data: DataFrame containing unemployment data.
- index_row: The row index that holds unemployment percentage data.
- year: The year of the data (e.g., 2016 or 2020).
"""
def extract_unemployment(data, index_row, year):
    percent_columns = [col for col in data.columns if 'County' in col and '!!Percent' in col]
    unemployment_data = data.loc[index_row, percent_columns]
    unemployment_data.index = [col.split('County')[0].strip() for col in percent_columns]
    unemployment_data_df = unemployment_data.reset_index().rename(columns={'index': 'County', index_row: 'Unemployment Rate'})
    unemployment_data_df['Year'] = year
    unemployment_data_df = unemployment_data_df[~unemployment_data_df['Unemployment Rate'].str.contains('±')]
    return unemployment_data_df

unem2016_extracted = extract_unemployment(unem2016, 5, 2016)

# Load and extract 2020 unemployment data
unem2020 = pd.read_csv('data/unemIncome2020.csv')
unem2020_extracted = extract_unemployment(unem2020, 5, 2020)

# Combine 2016 and 2020 unemployment data
combined_unemployment_data = pd.concat([unem2016_extracted, unem2020_extracted])
combined_unemployment_data.set_index(['County', 'Year'], inplace=True)
combined_unemployment_data = combined_unemployment_data.sort_index()

# Merge education and unemployment data
final_combined_data = pd.merge(combined_edu_data, combined_unemployment_data, how='outer', left_index=True, right_index=True)
final_combined_data.reset_index(inplace=True)

# Save to CSV
final_combined_data.to_csv('data/edunemployment.csv', index=False)


# Load and extract 2016 and 2020 income data
unemIncome2016 = pd.read_csv('data/unemIncome2016.csv')
unemIncome2020 = pd.read_csv('data/unemIncome2020.csv')

low_income_rows = list(range(57, 62))

"""
Converts a percentage string to a float rounded to one decimal place.
"""
def parse_percentage(value):
    try:
        return round(float(value.strip('%')), 1)
    except (ValueError, AttributeError):
        return None

"""
Extracts the percentage of individuals with income less than $50,000 for each county.

Parameters:
- df: DataFrame containing income data.
- year: The year of the data (e.g., 2016 or 2020).
"""
def extract_low_income_data(df, year):
    county_columns = [col for col in df.columns if 'County, Pennsylvania!!Percent' in col]
    county_data = {'County': [], 'Year': [], 'Income < 50000': []}
    
    for county_column in county_columns:
        low_income_sum = 0
        for row in low_income_rows:
            try:
                value = df.iloc[row, df.columns.get_loc(county_column)]
                if pd.notnull(value):
                    parsed_value = parse_percentage(value)
                    if parsed_value is not None:
                        low_income_sum += parsed_value
            except IndexError:
                continue
        county_name = county_column.split(',')[0].replace('County', '').strip()
        if low_income_sum > 0:
            county_data['County'].append(county_name)
            county_data['Year'].append(year)
            county_data['Income < 50000'].append(f"{low_income_sum:.1f}%")
    return pd.DataFrame(county_data).set_index(['County', 'Year'])

income_data_2016 = extract_low_income_data(unemIncome2016, 2016)
income_data_2020 = extract_low_income_data(unemIncome2020, 2020)

combined_income_data = pd.concat([income_data_2016, income_data_2020]).sort_index(level=['County', 'Year'])

combined_income_data.reset_index().to_csv('data/income_data.csv', index=False)

#combine income and eduunemp csv tgt
# Load existing education and unemployment data
edu_unemployment_data = pd.read_csv('data/edunemployment.csv')

# Load the newly created income data CSV
income_data = pd.read_csv('data/income_data.csv')

#COUNTY AREA

# The URL to submit the form
url = "https://www.rural.pa.gov/data/county-profiles.cfm"

# Form data
data = {
    'RDCategory': '15',  # '15' is the value for 'Square Land Miles, 2020'
    'Submit': 'Submit'   # This is the name of the submit button in the form
}

# Perform the POST request
response = requests.post(url, data=data)

# Check if the request was successful
if response.status_code == 200:
    # Parse the content with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the table with the class 'display'
    table = soup.find('table', {'class': 'display'})
    
    if table:
        # Initialize empty lists to store the county names and areas
        counties = []
        areas = []
        
        # Find all rows in the table (excluding the header)
        rows = table.find_all('tr')[1:]  # Skip the first header row
        
        for row in rows:
            cols = row.find_all('td')
            if len(cols) == 2:
                # Extract the county name (first column)
                county = cols[0].get_text(strip=True)
                
                # Split the county name and take the first part
                county_first_name = county.split()[0]  # Get the first word of the county name
                
                # Extract the area (second column)
                area = cols[1].get_text(strip=True)
                
                # Append to the lists
                counties.append(county_first_name)
                areas.append(area)
        
        # Create a DataFrame from the extracted data
        df = pd.DataFrame({
            'County': counties,
            'Square Land Miles': areas
        })
        
        # Save the DataFrame to a CSV file
        df.to_csv('data/countyArea.csv', index=False)
        
    else:
        print("Table with class 'display' not found.")
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")

# Load the CSV files into pandas DataFrames
county_area = pd.read_csv('data/countyArea.csv')

# Clean the 'County' column in both DataFrames for better matching
county_area['County'] = county_area['County'].str.strip()



# Load and extract 2016 male data
sex2016 = pd.read_csv('data/voterpopRace2016.csv')

"""
Extracts the male percentage data for each county.

Parameters:
- data: DataFrame containing sex data.
- index_row: The row index that holds the male percentage data.
- year: The year of the data (e.g., 2016 or 2020).
"""
def extract_male(data, index_row, year):
    percent_columns = [col for col in data.columns if 'County' in col and '!!Percent' in col]
    male2016_data = data.loc[index_row, percent_columns]
    male2016_data.index = [col.split('County')[0].strip() for col in percent_columns]
    male2016_data_df = male2016_data.reset_index().rename(columns={'index': 'County', index_row: 'Male Percentage'})
    male2016_data_df['Year'] = year
    male2016_data_df = male2016_data_df[~male2016_data_df['Male Percentage'].str.contains('±')]
    return male2016_data_df

male2016_extracted = extract_male(sex2016, 2, 2016)

# Load and extract 2020 male data
sex2020 = pd.read_csv('data/voterpopRace2020.csv')
male2020_extracted = extract_male(sex2020, 2, 2020)

# Combine 2016 and 2020 male data
combined_male_data = pd.concat([male2016_extracted, male2020_extracted])
combined_male_data.set_index(['County', 'Year'], inplace=True)
combined_male_data = combined_male_data.sort_index()

# Load and extract 2016 female data
sex2016 = pd.read_csv('data/voterpopRace2016.csv')

"""
Extracts the female percentage data for each county.

Parameters:
- data: DataFrame containing sex data.
- index_row: The row index that holds the female percentage data.
- year: The year of the data (e.g., 2016 or 2020).
"""
def extract_female(data, index_row, year):
    percent_columns = [col for col in data.columns if 'County' in col and '!!Percent' in col]
    female2016_data = data.loc[index_row, percent_columns]
    female2016_data.index = [col.split('County')[0].strip() for col in percent_columns]
    female2016_data_df = female2016_data.reset_index().rename(columns={'index': 'County', index_row: 'Female Percentage'})
    female2016_data_df['Year'] = year
    female2016_data_df = female2016_data_df[~female2016_data_df['Female Percentage'].str.contains('±')]
    return female2016_data_df

female2016_extracted = extract_female(sex2016, 3, 2016)

# Load and extract 2020 male data
sex2020 = pd.read_csv('data/voterpopRace2020.csv')
female2020_extracted = extract_female(sex2020, 3, 2020)

# Combine 2016 and 2020 female data
combined_female_data = pd.concat([female2016_extracted, female2020_extracted])
combined_female_data.set_index(['County', 'Year'], inplace=True)
combined_female_data = combined_female_data.sort_index()

# Merge male and female data
sex_data = pd.merge(combined_male_data, combined_female_data, how = 'outer', left_index=True, right_index=True)
sex_data.reset_index(inplace=True)
sex_data.to_csv('data/sex.csv')

# Step 1: Load the existing sex.csv
sex_data = pd.read_csv('data/sex.csv', usecols=['County', 'Year', 'Male Percentage', 'Female Percentage'])

# Step 2: Clean the `sex_data`
# Remove the '%' symbol and convert the columns to numeric
sex_data['Male Percentage'] = sex_data['Male Percentage'].str.replace('%', '').astype(float)
sex_data['Female Percentage'] = sex_data['Female Percentage'].str.replace('%', '').astype(float)

# Step 3: Merge the `sex_data` into the `final_combined_data`
# Strip whitespace from the 'County' column if necessary
sex_data['County'] = sex_data['County'].str.strip()



# Load and extract 2016 hispanic data
race12016 = pd.read_csv('data/voterpopRace2016.csv')

"""
Extracts the Hispanic percentage data for each county.

Parameters:
- data: DataFrame containing race data.
- index_row: The row index that holds the Hispanic percentage data.
- year: The year of the data (e.g., 2016 or 2020).
"""
def extract_hispanic(data, index_row, year):
    percent_columns = [col for col in data.columns if 'County' in col and '!!Percent' in col]
    hispanic2016_data = data.loc[index_row, percent_columns]
    hispanic2016_data.index = [col.split('County')[0].strip() for col in percent_columns]
    hispanic2016_data_df = hispanic2016_data.reset_index().rename(columns={'index': 'County', index_row: 'Hispanic percentage'})
    hispanic2016_data_df['Year'] = year
    hispanic2016_data_df = hispanic2016_data_df[hispanic2016_data_df['Hispanic percentage'].apply(lambda x: x.replace('%', '').replace('.', '', 1).isdigit())]
    return hispanic2016_data_df

hispanic2016_extracted = extract_hispanic(race12016, 69, 2016)

# Load and extract 2020 hispanic data
race2020 = pd.read_csv('data/voterpopRace2020.csv')
hispanic2020_extracted = extract_hispanic(race2020, 74, 2020)

# Combine 2016 and 2020 hispanic data
combined_hispanic_data = pd.concat([hispanic2016_extracted, hispanic2020_extracted])
combined_hispanic_data.set_index(['County', 'Year'], inplace=True)
combined_hispanic_data = combined_hispanic_data.sort_index()





# Load and extract 2016 black data
race2016 = pd.read_csv('data/voterpopRace2016.csv')

"""
Extracts the Black percentage data for each county.

Parameters:
- data: DataFrame containing race data.
- index_row: The row index that holds the Black percentage data.
- year: The year of the data (e.g., 2016 or 2020).
"""
def extract_black(data, index_row, year):
    percent_columns = [col for col in data.columns if 'County' in col and '!!Percent' in col]
    black2016_data = data.loc[index_row, percent_columns]
    black2016_data.index = [col.split('County')[0].strip() for col in percent_columns]
    black2016_data_df = black2016_data.reset_index().rename(columns={'index': 'County', index_row: 'Black percentage'})
    black2016_data_df['Year'] = year
    black2016_data_df = black2016_data_df[black2016_data_df['Black percentage'].apply(lambda x: x.replace('%', '').replace('.', '', 1).isdigit())]
    return black2016_data_df

black2016_extracted = extract_black(race2016, 34, 2016)

# Load and extract 2020 black data
race2020 = pd.read_csv('data/voterpopRace2020.csv')
black2020_extracted = extract_black(race2020, 39, 2020)

# Combine 2016 and 2020 black data
combined_black_data = pd.concat([black2016_extracted, black2020_extracted])
combined_black_data.set_index(['County', 'Year'], inplace=True)
combined_black_data = combined_black_data.sort_index()





# Load and extract 2016 Native American data
race12016 = pd.read_csv('data/voterpopRace2016.csv')

"""
Extracts the Native American percentage data for each county.

Parameters:
- data: DataFrame containing race data.
- index_row: The row index that holds the Native American percentage data.
- year: The year of the data (e.g., 2016 or 2020).
"""
def extract_nativeamerican(data, index_row, year):
    percent_columns = [col for col in data.columns if 'County' in col and '!!Percent' in col]
    nativeamerican2016_data = data.loc[index_row, percent_columns]
    nativeamerican2016_data.index = [col.split('County')[0].strip() for col in percent_columns]
    nativeamerican2016_data_df = nativeamerican2016_data.reset_index().rename(columns={'index': 'County', index_row: 'Native American percentage'})
    nativeamerican2016_data_df['Year'] = year
    
    # Filter out rows where the 'nativeamerican percentage' contains non-numeric values
    nativeamerican2016_data_df = nativeamerican2016_data_df[nativeamerican2016_data_df['Native American percentage'].apply(lambda x: x.replace('%', '').replace('.', '', 1).isdigit())]
    
    return nativeamerican2016_data_df

nativeamerican2016_extracted = extract_nativeamerican(race12016, 35, 2016)

# Load and extract 2020 Native American data
race2020 = pd.read_csv('data/voterpopRace2020.csv')
nativeamerican2020_extracted = extract_nativeamerican(race2020, 40, 2020)

# Combine 2016 and 2020 Native American data
combined_nativeamerican_data = pd.concat([nativeamerican2016_extracted, nativeamerican2020_extracted])
combined_nativeamerican_data.set_index(['County', 'Year'], inplace=True)
combined_nativeamerican_data = combined_nativeamerican_data.sort_index()






# Load and extract 2016 Asian data
race12016 = pd.read_csv('data/voterpopRace2016.csv')

"""
Extracts the Asian percentage data for each county.

Parameters:
- data: DataFrame containing race data.
- index_row: The row index that holds the Asian percentage data.
- year: The year of the data (e.g., 2016 or 2020).
"""
def extract_asian(data, index_row, year):
    percent_columns = [col for col in data.columns if 'County' in col and '!!Percent' in col]
    asian2016_data = data.loc[index_row, percent_columns]
    asian2016_data.index = [col.split('County')[0].strip() for col in percent_columns]
    asian2016_data_df = asian2016_data.reset_index().rename(columns={'index': 'County', index_row: 'Asian percentage'})
    asian2016_data_df['Year'] = year
    
    # Filter out rows where the 'asian percentage' contains non-numeric values
    asian2016_data_df = asian2016_data_df[asian2016_data_df['Asian percentage'].apply(lambda x: x.replace('%', '').replace('.', '', 1).isdigit())]
    
    return asian2016_data_df

asian2016_extracted = extract_asian(race12016, 40, 2016)

# Load and extract 2020 Asian data
race2020 = pd.read_csv('data/voterpopRace2020.csv')
asian2020_extracted = extract_asian(race2020, 45, 2020)

# Combine 2016 and 2020 Asian data
combined_asian_data = pd.concat([asian2016_extracted, asian2020_extracted])
combined_asian_data.set_index(['County', 'Year'], inplace=True)
combined_asian_data = combined_asian_data.sort_index()





# Load and extract 2016 Hawaiian data
race12016 = pd.read_csv('data/voterpopRace2016.csv')

"""
Extracts the Hawaiian percentage data for each county.

Parameters:
- data: DataFrame containing race data.
- index_row: The row index that holds the Hawaiian percentage data.
- year: The year of the data (e.g., 2016 or 2020).
"""
def extract_hawaiian(data, index_row, year):
    percent_columns = [col for col in data.columns if 'County' in col and '!!Percent' in col]
    hawaiian2016_data = data.loc[index_row, percent_columns]
    hawaiian2016_data.index = [col.split('County')[0].strip() for col in percent_columns]
    hawaiian2016_data_df = hawaiian2016_data.reset_index().rename(columns={'index': 'County', index_row: 'Hawaiian percentage'})
    hawaiian2016_data_df['Year'] = year
    
    # Filter out rows where the 'hawaiian percentage' contains non-numeric values
    hawaiian2016_data_df = hawaiian2016_data_df[hawaiian2016_data_df['Hawaiian percentage'].apply(lambda x: x.replace('%', '').replace('.', '', 1).isdigit())]
    
    return hawaiian2016_data_df

hawaiian2016_extracted = extract_hawaiian(race12016, 48, 2016)

# Load and extract 2020 Hawaiian data
race2020 = pd.read_csv('data/voterpopRace2020.csv')
hawaiian2020_extracted = extract_hawaiian(race2020, 53, 2020)

# Combine 2016 and 2020 Hawaiian data
combined_hawaiian_data = pd.concat([hawaiian2016_extracted, hawaiian2020_extracted])
combined_hawaiian_data.set_index(['County', 'Year'], inplace=True)
combined_hawaiian_data = combined_hawaiian_data.sort_index()






# Load and extract 2016 Other Races data
race12016 = pd.read_csv('data/voterpopRace2016.csv')

"""
Extracts the Other Races percentage data for each county.

Parameters:
- data: DataFrame containing race data.
- index_row: The row index that holds the Other Races percentage data.
- year: The year of the data (e.g., 2016 or 2020).
"""
def extract_otherRaces(data, index_row, year):
    percent_columns = [col for col in data.columns if 'County' in col and '!!Percent' in col]
    otherRaces2016_data = data.loc[index_row, percent_columns]
    otherRaces2016_data.index = [col.split('County')[0].strip() for col in percent_columns]
    otherRaces2016_data_df = otherRaces2016_data.reset_index().rename(columns={'index': 'County', index_row: 'Other Races percentage'})
    otherRaces2016_data_df['Year'] = year
    
    # Filter out rows where the 'otherRaces percentage' contains non-numeric values
    otherRaces2016_data_df = otherRaces2016_data_df[otherRaces2016_data_df['Other Races percentage'].apply(lambda x: x.replace('%', '').replace('.', '', 1).isdigit())]
    
    return otherRaces2016_data_df

otherRaces2016_extracted = extract_otherRaces(race12016, 53, 2016)

# Load and extract 2020 Other Races data
race2020 = pd.read_csv('data/voterpopRace2020.csv')
otherRaces2020_extracted = extract_otherRaces(race2020, 58, 2020)

# Combine 2016 and 2020 Other Races data
combined_otherRaces_data = pd.concat([otherRaces2016_extracted, otherRaces2020_extracted])
combined_otherRaces_data.set_index(['County', 'Year'], inplace=True)
combined_otherRaces_data = combined_otherRaces_data.sort_index()





# Load and extract 2016 More Than One Races data
race12016 = pd.read_csv('data/voterpopRace2016.csv')


"""
Extracts the More Than One Race percentage data for each county.

Parameters:
- data: DataFrame containing race data.
- index_row: The row index that holds the More Than One Race percentage data.
- year: The year of the data (e.g., 2016 or 2020).
"""
def extract_more_than_oneRaces(data, index_row, year):
    percent_columns = [col for col in data.columns if 'County' in col and '!!Percent' in col]
    more_than_oneRaces2016_data = data.loc[index_row, percent_columns]
    more_than_oneRaces2016_data.index = [col.split('County')[0].strip() for col in percent_columns]
    more_than_oneRaces2016_data_df = more_than_oneRaces2016_data.reset_index().rename(columns={'index': 'County', index_row: 'More Than One Races percentage'})
    more_than_oneRaces2016_data_df['Year'] = year
    
    # Filter out rows where the 'more_than_oneRaces percentage' contains non-numeric values
    more_than_oneRaces2016_data_df = more_than_oneRaces2016_data_df[more_than_oneRaces2016_data_df['More Than One Races percentage'].apply(lambda x: x.replace('%', '').replace('.', '', 1).isdigit())]
    
    return more_than_oneRaces2016_data_df

more_than_oneRaces2016_extracted = extract_more_than_oneRaces(race12016, 54, 2016)

# Load and extract 2020 More Than One Races data
race2020 = pd.read_csv('data/voterpopRace2020.csv')
more_than_oneRaces2020_extracted = extract_more_than_oneRaces(race2020, 59, 2020)

# Combine 2016 and 2020 More Than One Races data
combined_more_than_oneRaces_data = pd.concat([more_than_oneRaces2016_extracted, more_than_oneRaces2020_extracted])
combined_more_than_oneRaces_data.set_index(['County', 'Year'], inplace=True)
combined_more_than_oneRaces_data = combined_more_than_oneRaces_data.sort_index()



# Merge all races data
race_data = pd.merge(combined_hispanic_data, combined_black_data, how='outer', left_index=True, right_index=True)
race_data = pd.merge(race_data, combined_nativeamerican_data, how='outer', left_index=True, right_index=True)
race_data = pd.merge(race_data, combined_asian_data, how='outer', left_index=True, right_index=True)
race_data = pd.merge(race_data, combined_hawaiian_data, how='outer', left_index=True, right_index=True)
race_data = pd.merge(race_data, combined_otherRaces_data, how='outer', left_index=True, right_index=True)
race_data = pd.merge(race_data, combined_more_than_oneRaces_data, how='outer', left_index=True, right_index=True)

# Reset index and export to CSV
race_data.reset_index(inplace=True)
race_data.to_csv('data/race.csv')

#EXTRACT RACE AREA
race_data = pd.read_csv('data/race.csv') 

#Voter Results
#Party Date
#2020
url = 'https://www.electionreturns.pa.gov/_ENR/api/ElectionReturn/GetCountyBreak?officeId=1&districtId=1&methodName=GetCountyBreak&electionid=83&electiontype=G&isactive=0'
response = requests.get(url)

resp = eval(eval(response.content))
infos = resp['Election']['Statewide'][0]

counties = list(infos.keys())
cols = list(infos[counties[0]][0].keys())

dct = {col: [] for col in cols}
for county in counties:
    for candidate in infos[county]:
        for col in cols:
            try:
                dct[col].append(candidate[col])
            except Exception as ex:
                dct[col].append('')

data = pd.DataFrame(dct)
df_2020_pivot = data.pivot_table(
    index='CountyName',
    columns='CandidateName',
    values=['Votes', 'ElectionDayVotes', 'MailInVotes', 'ProvisionalVotes','Percentage'],
    aggfunc='sum').reset_index()

df_2020_pivot.columns = ['_'.join(col).strip() if type(col) is tuple else col for col in df_2020_pivot.columns]

df_2020_pivot.columns.name = None
df_2020_pivot.rename(columns={"CountyName": "County"}, inplace=True)
df_2020_pivot['Total Votes'] = df_2020_pivot.iloc[:, -3:].astype(float).sum(axis=1)

df_2020_pivot.to_csv('data/transformed_2020.csv', index=False)

voterResult2020 = pd.read_csv('data/transformed_2020.csv')
voterResult2020_selected = voterResult2020[['CountyName_', 'Total Votes']].copy()
voterResult2020_selected.rename(columns={'CountyName_': 'County'}, inplace=True)
voterResult2020_selected['County'] = voterResult2020_selected['County'].str.title()
voterResult2020_selected['County'] = voterResult2020_selected['County'].replace('Mckean', 'McKean')
voterResult2020_selected['Year'] = 2020

#2016
url = 'https://www.electionreturns.pa.gov/_ENR/api/ElectionReturn/GetCountyBreak?officeId=1&districtId=1&methodName=GetCountyBreak&electionid=54&electiontype=G&isactive=0'
response = requests.get(url)

resp = eval(eval(response.content))
infos = resp['Election']['Statewide'][0]

counties = list(infos.keys())
cols = list(infos[counties[0]][0].keys())

dct = {col: [] for col in cols}
for county in counties:
    for candidate in infos[county]:
        for col in cols:
            try:
                dct[col].append(candidate[col])
            except Exception as ex:
                dct[col].append('')

data = pd.DataFrame(dct)

df_2016_pivot = data.pivot_table(
    index='CountyName',
    columns='CandidateName',
    values=['Votes', 'Percentage'],
    aggfunc='sum').reset_index()

df_2016_pivot.columns = ['_'.join(col).strip() if type(col) is tuple else col for col in df_2016_pivot.columns]

df_2016_pivot.columns.name = None
df_2016_pivot.rename(columns={"CountyName": "County"}, inplace=True)
df_2016_pivot['Total Votes'] = df_2016_pivot.iloc[:, -5:].astype(float).sum(axis=1)

df_2016_pivot.to_csv('data/transformed_2016.csv', index=False)

voterResult2016 = pd.read_csv('data/transformed_2016.csv')
voterResult2016_selected = voterResult2016[['CountyName_', 'Total Votes']].copy()
voterResult2016_selected.rename(columns={'CountyName_': 'County'}, inplace=True)
voterResult2016_selected['County'] = voterResult2016_selected['County'].str.title()
voterResult2016_selected['County'] = voterResult2016_selected['County'].replace('Mckean', 'McKean')
voterResult2016_selected['Year'] = 2016

#Diversity Index
diversityIndex = pd.read_csv("data/diversityIndex.csv")
diversity_data = diversityIndex[['County', '2020']].copy()
diversity_data.rename(columns={'2020': 'Diversity Index 2020'}, inplace=True)
diversity_data['County'] = diversity_data['County'].str.strip()

#Population of voters (age above 18)
# 2020

data = pd.read_csv('data/voterpopRace2020.csv')
data.index = data['Label (Grouping)']

column_name = data.columns.tolist()
r = 0
new_column = []
for i in column_name:
    if re.search(r'County, Pennsylvania!!Estimate', i):
        new_column.append(i)

temp = []
pop18 = []
for c in new_column:
    temp.append(c)
    pop18.append(int(''.join(data.iloc[91][c].split(','))))

county = []
for i in temp:
   county.append(i.split(' ')[0])

pop_18 = pd.DataFrame({'County': county, 'pop18': pop18})

pop_18.to_csv('data/pop_18_2020.csv', index=False)

# 2016
data = pd.read_csv('data/voterpopRace2016.csv')
data.index = data['Label (Grouping)']
column_name = data.columns.tolist()

new_column = []
for i in column_name:
    if re.search(r'County, Pennsylvania!!Estimate', i):
        new_column.append(i)

temp = []
pop18 = []
for c in new_column:
    temp.append(c)
    pop18.append(int(''.join(data.iloc[86][c].split(','))))

county = []
for i in temp:
   county.append(i.split(' ')[0])

pop_18 = pd.DataFrame({'County': county,
                                'pop18': pop18})

pop_18.to_csv('data/pop_18_2016.csv', index=False)

# combine
data_16 = pd.read_csv('data/pop_18_2016.csv')
data_16.index = data_16['County']
data_20 = pd.read_csv('data/pop_18_2020.csv')
data_20.index = data_20['County']

population_data = pd.DataFrame({'County': [], 'year': [], 'pop18': []}) 

for d16 in data_16['County']:
    for d20 in data_20['County']:
        if d16 == d20:
            newRow = pd.DataFrame({'County':[d16, d20],'year':[2016, 2020], 'pop18':[data_16.loc[d16]['pop18'], data_20.loc[d20]['pop18']]})
            population_data = pd.concat([population_data, newRow])

population_data.to_csv('data/pop_18.csv', index=False)
population_data.rename(columns={'pop18':'Population', 'year': 'Year'}, inplace=True)

#TOTAL Population
totalPop2016 = pd.read_csv('data/voterpopRace2016.csv')

"""
Extracts the total population estimates for each county.

Parameters:
- data: DataFrame containing total population data.
- index_row: The row index that holds the total population estimate.
- year: The year of the data (e.g., 2016 or 2020).
"""
def extract_pop(data, index_row, year):
    percent_columns = [col for col in data.columns if 'County' in col and '!!Estimate' in col]
    pop2016_data = data.loc[index_row, percent_columns]
    pop2016_data.index = [col.split('County')[0].strip() for col in percent_columns]
    pop2016_data_df = pop2016_data.reset_index().rename(columns={'index': 'County', index_row: 'Total Pop percentage'})
    pop2016_data_df['Year'] = year
    return pop2016_data_df

pop2016_extracted = extract_pop(totalPop2016, 1, 2016)

# Load and extract 2020 total population data
pop2020 = pd.read_csv('data/voterpopRace2020.csv')
pop2020_extracted = extract_pop(pop2020, 1, 2020)

# Combine 2016 and 2020 total population data
combined_pop_data = pd.concat([pop2016_extracted, pop2020_extracted])
combined_pop_data.set_index(['County', 'Year'], inplace=True)
combined_pop_data = combined_pop_data.sort_index()

combined_pop_data.to_csv('data/totalPop.csv')
total_population_data = pd.read_csv('data/totalPop.csv')


#Partisan
election_2012 = pd.read_excel('data/transformed_2012.xlsx')
election_2016 = pd.read_csv('data/transformed_2016.csv')
election_2020 = pd.read_csv('data/transformed_2020.csv')

# Step 1: Merge edu_unemployment_data and income_data
merged_data = pd.merge(edu_unemployment_data, income_data, on=['County', 'Year'], how='outer')

# Step 2: Merge the result with county_area
merged_data = pd.merge(merged_data, county_area, on=['County'], how='outer')

# Step 3: Merge the result with race data
merged_data = pd.merge(merged_data, race_data, on=['County', 'Year'], how='outer' )

# Step 4: Merge the result with sex_data
merged_data = pd.merge(merged_data, sex_data, on=['County', 'Year'], how='outer')

# Step 5: Merge the result with diversity
merged_data = pd.merge(merged_data, diversity_data, on=['County'], how='outer')

# Step 6: Merge the result with population
merged_data = pd.merge(merged_data, population_data, on=['County', 'Year'], how='outer')

#Step 7: Merge result with total population
merged_data = pd.merge(merged_data, total_population_data, on=['County', 'Year'], how='outer')

# Step 8: Merge the final result with voter results
merged_data = pd.merge(merged_data, voterResult2016_selected, on=['County', 'Year'], how='outer')

FinalCombinedData = pd.merge(merged_data, voterResult2020_selected, on=['County', 'Year'], how='outer')
FinalCombinedData['Total_Votes'] = FinalCombinedData['Total Votes_x'].fillna(0) + FinalCombinedData['Total Votes_y'].fillna(0)

# Optionally, drop the original vote columns if you only want the total
FinalCombinedData = FinalCombinedData.drop(columns=['Total Votes_x', 'Total Votes_y'])
FinalCombinedData = FinalCombinedData[:-1]
FinalCombinedData['Year'] = FinalCombinedData['Year'].astype(int)

# Save the final combined data to CSV
FinalCombinedData.to_csv('data/FinalCombinedData.csv', index=False)

data = pd.read_csv('data/FinalCombinedData.csv')

data.rename(columns={
    'High School': '% High School', "Bachelor's or Higher": "% Bachelor's or Higher",
    'Unemployment Rate': '% Unemployment Rate', 'Income < 50000': '% Income < 50000',
    'Hispanic percentage': '% Hispanic', 'Black percentage': '% Black',
    'Native American percentage': '% Native American', 'Asian percentage': '% Asian',
    'Hawaiian percentage': '% Hawaiian', 'Other Races percentage': '% Other Races',
    'More Than One Races percentage': '% More Than One Races', 'Male Percentage': '% Male',
    'Female Percentage': "% Female", 'Diversity Index 2020': '% Diversity Index 2020',
    'Population': 'Eligible Voters', 'Total Pop percentage': 'Total Population'
}, inplace=True)

column_name = ['% High School', "% Bachelor's or Higher", '% Unemployment Rate',
               '% Income < 50000', '% Hispanic', '% Black', '% Native American',
               '% Asian', '% Hawaiian', '% Other Races', '% More Than One Races', '% Diversity Index 2020']

# 
def fill_null(name):
    for idx, item in enumerate(data[name]):
        if type(item) == float:
            if idx > 0:
                if data.loc[idx - 1, 'County'] == data.loc[idx, 'County']:
                    data.loc[idx, name] = data.loc[idx - 1, name]
                elif data.loc[idx + 1, 'County'] == data.loc[idx, 'County']:
                    data.loc[idx, name] = data.loc[idx + 1, name]

def clean_percentage(name):
    for idx, item in enumerate(data[name]):
        if type(item) == float:
            print(data.loc[idx, 'County'], item, name)
        else:
            data.loc[idx, name] = float(item.split('%')[0]) / 100

def for_sex(name):
    for idx, item in enumerate(data[name]):
        data.loc[idx, name] = float(item) / 100

for name in column_name:
    fill_null(name=name)

for name in column_name:
    clean_percentage(name=name)

for item in ['% Male', '% Female']:
    for_sex(item)

data = data.drop('Unnamed: 0', axis=1)

#Adding Partisan Data
for idx1, county1 in enumerate(election_2012['CountyName_']):
    for idx2, county2 in enumerate(data['County']):
        if (county1.upper() == county2.upper()) and (data.loc[idx2, 'Year'] == 2016):
            data.loc[idx2, 'Prev Democratic Vote Share'] = float(election_2012.loc[idx1, 'Percentage_OBAMA, BARACK']) / 100

for idx1, county1 in enumerate(election_2016['CountyName_']):
    for idx2, county2 in enumerate(data['County']):
        if (county1.upper() == county2.upper()) and (data.loc[idx2, 'Year'] == 2020):
            data.loc[idx2, 'Prev Democratic Vote Share'] = float(election_2016.loc[idx1, 'Percentage_CLINTON, HILLARY']) / 100

data.to_csv('data/cleanedData.csv', index=False)
print("Final Combined and Cleaned CSV Created Successfully.")