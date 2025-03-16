import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import geopandas as gpd
import kaleido
import os

# Create images folder if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

# Load the CSV file
file_path = 'data/cleanedData.csv'
data = pd.read_csv(file_path)
data['Total Population'] = data['Total Population'].str.replace(',', '').astype(float)
# Voter Turnout
data['Voter Turnout'] = data['Total_Votes'] / data['Eligible Voters']

# Separate the combined data into 2016 and 2020 dataframes
data_2016 = data[data['Year'] == 2016]
data_2020 = data[data['Year'] == 2020]

# Voter Turnout Increase/Decrease for All Counties
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x='Year', y='Voter Turnout', hue='County', marker='o')
plt.title('Voter Turnout Increase/Decrease for All Counties')
plt.ylabel('Voter Turnout')
plt.xlabel('Year')
plt.xticks([2016, 2020])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.subplots_adjust(bottom=0.15, top=0.85)
plt.savefig('images/voter_turnout_increase_decrease_all_counties.png')
plt.show()

# Calculate population density
valid_data = data.copy()
valid_data['Total Population'] = pd.to_numeric(valid_data['Total Population'], errors='coerce')
valid_data['Square Land Miles'] = pd.to_numeric(valid_data['Square Land Miles'], errors='coerce')
valid_data['Population Density'] = valid_data['Total Population'] / valid_data['Square Land Miles']

# Urban/Rural
valid_data['Urban/Rural'] = valid_data['Population Density'].apply(lambda x: 'Urban' if x > 100 else 'Rural')

# Voter Turnout Rate by County and Urban/Rural Classification
plt.figure(figsize=(12, 6))
sns.scatterplot(data=valid_data, x='County', y='Voter Turnout', hue='Urban/Rural', style='Year')
plt.title('Voter Turnout Rate by County and Urban/Rural Classification')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('images/voter_turnout_by_county_urban_rural.png')
plt.show()

# Pair plot of key variables
pairplot_data = valid_data[['Voter Turnout', 'Population Density', '% High School', "% Bachelor's or Higher", '% Unemployment Rate', 'Year']].copy()
pairplot_data_clean = pairplot_data.dropna()
sns.pairplot(pairplot_data_clean, hue='Year', diag_kind='kde')
plt.savefig('images/pairplot_voter_turnout_variables.png')
plt.show()

# Voter Turnout by County (2016 AND 2020) with maps
# Load shapefile for US counties
gdf = gpd.read_file('https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_20m.zip')

# Filter for Pennsylvania counties (FIPS code for Pennsylvania is '42')
gdf_pa = gdf[gdf['STATEFP'] == '42']

# Merge shapefiles with 2016 and 2020 data
gdf_pa_2016 = gdf_pa.merge(data_2016, how='left', left_on='NAME', right_on='County')
gdf_pa_2020 = gdf_pa.merge(data_2020, how='left', left_on='NAME', right_on='County')

# Plot Voter Turnout for 2016 using Plotly
fig_2016 = px.choropleth(gdf_pa_2016, 
                         geojson=gdf_pa_2016.geometry, 
                         locations=gdf_pa_2016.index, 
                         color='Voter Turnout',
                         hover_name='County',
                         title='Voter Turnout by County - Pennsylvania (2016)',
                         color_continuous_scale="Viridis")
fig_2016.update_geos(fitbounds="locations", visible=False)
fig_2016.update_layout(legend_title_text='Voter Turnout (%)')
fig_2016.write_image('images/voter_turnout_pa_2016.png')  # Save as image

# Plot Voter Turnout for 2020 using Plotly
fig_2020 = px.choropleth(gdf_pa_2020, 
                         geojson=gdf_pa_2020.geometry, 
                         locations=gdf_pa_2020.index, 
                         color='Voter Turnout',
                         hover_name='County',
                         title='Voter Turnout by County - Pennsylvania (2020)',
                         color_continuous_scale="Blues")
fig_2020.update_geos(fitbounds="locations", visible=False)
fig_2020.update_layout(legend_title_text='Voter Turnout (%)')
fig_2020.write_image('images/voter_turnout_pa_2020.png')  # Save as image

# Show the interactive maps
fig_2016.show()
fig_2020.show()

# Plot Democratic Votes for 2012
fig_dem_votes_2016 = px.choropleth(gdf_pa_2016, 
                                   geojson=gdf_pa_2016.geometry, 
                                   locations=gdf_pa_2016.index, 
                                   color='Prev Democratic Vote Share',
                                   hover_name='County',
                                   title='Democratic Votes by County - Pennsylvania (2012)',
                                   color_continuous_scale="Viridis")
fig_dem_votes_2016.update_geos(fitbounds="locations", visible=False)
fig_dem_votes_2016.update_layout(legend_title_text='Democratic Votes (2012)')
fig_dem_votes_2016.write_image('images/democratic_votes_pa_2012.png')  # Save as image

# Plot Democratic Votes for 2016
fig_dem_votes_2020 = px.choropleth(gdf_pa_2020, 
                                   geojson=gdf_pa_2020.geometry, 
                                   locations=gdf_pa_2020.index, 
                                   color='Prev Democratic Vote Share',
                                   hover_name='County',
                                   title='Democratic Votes by County - Pennsylvania (2016)',
                                   color_continuous_scale="Blues")
fig_dem_votes_2020.update_geos(fitbounds="locations", visible=False)
fig_dem_votes_2020.update_layout(legend_title_text='Democratic Votes (2016)')
fig_dem_votes_2020.write_image('images/democratic_votes_pa_2016.png')  # Save as image

# Show the interactive maps
fig_dem_votes_2016.show()
fig_dem_votes_2020.show()

# Racial Composition and Voter Turnout Comparison (2016 and 2020)
racial_cols = ['% Hispanic', '% Black', '% Native American', '% Asian', '% Hawaiian', '% Other Races', '% More Than One Races']
voter_turnout_col = 'Voter Turnout'
pastel_colors = ['#aed6f1', '#a3e4d7', '#f9e79f', '#f5b7b1', '#d7bde2', '#fad7a0', '#a2d9ce'] 
counties = data_2020['County'].unique()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8))

# 2016 data
cumulative_2016 = np.zeros(len(data_2016))
for i, race in enumerate(racial_cols):
    ax1.bar(data_2016['County'], data_2016[race], bottom=cumulative_2016, label=race, color=pastel_colors[i])
    cumulative_2016 += data_2016[race]

# Add voter turnout line for 2016
ax1_twin = ax1.twinx()
ax1_twin.plot(data_2016['County'], data_2016[voter_turnout_col], color='pink', marker='o', label='Voter Turnout')

# 2020 data
cumulative_2020 = np.zeros(len(data_2020))
for i, race in enumerate(racial_cols):
    ax2.bar(data_2020['County'], data_2020[race], bottom=cumulative_2020, label=race, color=pastel_colors[i])
    cumulative_2020 += data_2020[race]

# Add voter turnout line for 2020
ax2_twin = ax2.twinx()
ax2_twin.plot(data_2020['County'], data_2020[voter_turnout_col], color='pink', marker='o', label='Voter Turnout')

# Titles and axis labels
ax1.set_title('Racial Composition and Voter Turnout in 2016')
ax1.set_ylabel('Racial Composition (%)')
ax1_twin.set_ylabel('Voter Turnout (%)')
ax1.set_xticks(np.arange(len(data_2016['County'])))
ax1.set_xticklabels(data_2016['County'], rotation=90)

ax2.set_title('Racial Composition and Voter Turnout in 2020')
ax2.set_ylabel('Racial Composition (%)')
ax2_twin.set_ylabel('Voter Turnout (%)')
ax2.set_xticks(np.arange(len(data_2020['County'])))  
ax2.set_xticklabels(data_2020['County'], rotation=90)

# Plot title for the entire figure
fig.suptitle('Racial Composition and Voter Turnout Comparison (2016 and 2020)', fontsize=16)

# Add legends
ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax1_twin.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax2_twin.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

# Adjust layout and save the figure
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('images/racial_composition_voter_turnout_comparison.png')
plt.show()



cols_to_convert = data.columns.difference(['County', 'Year'])
data[cols_to_convert] = data[cols_to_convert].apply(pd.to_numeric, errors='coerce')

# Group by 'County' and calculate the mean for numeric columns (excluding 'County')
aggregated_data = data.groupby('County')[cols_to_convert].mean().reset_index()

# Ensure that the resulting aggregated_data contains only numeric values (drop non-numeric columns)
numeric_cols = aggregated_data.select_dtypes(include=[np.number]).columns
aggregated_numeric_data = aggregated_data[numeric_cols]

# Calculate the correlation matrix of the numeric data
county_corr = aggregated_numeric_data.corr()

# Set up the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(county_corr, annot=True, fmt=".2f", cmap='PuBuGn', square=True, cbar_kws={"shrink": .8})

# Add title and save the figure
plt.title('Aggregated County Correlation Heatmap')
plt.savefig('images/county_correlation_heatmap.png')
plt.show()
