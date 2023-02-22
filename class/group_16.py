import pandas as pd
import requests
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from string import ascii_letters

## This is the python file for our class that will have several methods
class Agros:

## Innit method:
################
    def __init__(self):
        self.data = None

## Method 1:
############

    def data_setup(self):
        #1. We get the path of our new directory downloads we will be making by adding the path of this .py file and "/downloads"
        absolute_path = os.path.dirname(__file__)
        relative_path = "/downloads"
        full_path = absolute_path + relative_path 

        #2. But we're only allowed to make this new directory once. So we check before making if /downloads already exists. 
        # If it does not exist, we make this downloads directory using the function os.mkdir()
        if not os.path.exists(full_path):
            os.mkdir(full_path)

        #3. This is the url where our dataset is located online
        url = "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/Agricultural%20total%20factor%20productivity%20(USDA)/Agricultural%20total%20factor%20productivity%20(USDA).csv"
        #4. We download the info from this url using the requests library and put the response.text in a csv file dataset.csv that we create in the downloads dir
        response = requests.get(url)
        file_path = os.path.join(full_path,'dataset.csv')
        with open(file_path, "w") as f:
            f.write(response.text)

        #5. Read the dataset.csv file into a pandas dataframe and make it an attribute of our class (self.data)
        #6. Only take into account data after 1970 (1970 included)
        dataframe = pd.read_csv(file_path)
        self.data = dataframe



## Method 2:
############

# Show a list of all available countries in the dataset


## Method 3:
############

#Plot an area chart of consumption (columns biofuel_consumption, coal_consumption, fossil_fuel_consumption, gas_consumption, 
# hydro_consumption, low_carbon_consumption, nuclear_consumption, oil_consumption, other_renewable_consumption, primary_energy_consumption, 
# renewables_consumption, solar_consumption, wind_consumption)
    def correlate__quantities(self):
        # Correlation between different variables
        corr = self.data[["output_quantity","crop_output_quantity","animal_output_quantity","fish_output_quantity","ag_land_quantity","labor_quantity",
                         "capital_quantity","machinery_quantity","livestock_quantity","fertilizer_quantity","animal_feed_quantity","cropland_quantity",
                         "pasture_quantity","irrigation_quantity"]].corr()
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})


## Method 4:
############

#input: a country or a list of countries (in string format)

#compare the total of the "_consumption" columns for each of the chosen countries

# plot it


## Method 5:
############

#input: a country or a list of countries (in string format)

#compare the "gdp" column of each country over the years


## Method 6:
############

#called gapminder

#input: a year (in integer format)
#If it is not an integer, the method should raise a TypeError

# Scatter plot where x is fertilizer_quantity, y is output_quantity, and the area of each dot should be a third relevant variable you find with exploration of the data.


# Calling of the methods
########################
object_1 = Agros()
object_1.data_setup()
object_1.correlate__quantities()
