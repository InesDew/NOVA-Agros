<<<<<<< Updated upstream
=======
import pandas as pd
import requests
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from string import ascii_letters

>>>>>>> Stashed changes
## This is the python file for our class that will have several methods
class MyClass:

## Innit method:
################
def __init__(self):
    self.data = None

## Method 1:
############
 
# Download the data file into a /downloads directory
 
#But only do this once

# Read the dataset into a pandas dataframe which is an attribute of your class (self.data)
def method_1(self):
    dataframe = read(dataset)
    self.data = dataframe


## Method 2:
############

# Show a list of all available countries in the dataset
output = list(countries)


## Method 3:
############

#Plot an area chart of consumption (columns biofuel_consumption, coal_consumption, fossil_fuel_consumption, gas_consumption, 
# hydro_consumption, low_carbon_consumption, nuclear_consumption, oil_consumption, other_renewable_consumption, primary_energy_consumption, 
# renewables_consumption, solar_consumption, wind_consumption)

<<<<<<< Updated upstream
#needs a country argument and a normalize argument

#return a ValueError when the chosen country does not exist
=======
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
>>>>>>> Stashed changes


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

<<<<<<< Updated upstream
#scatter plot where x is gdp, y is total energy consumption, and the area of each dot is population.
def gapminder(year:int):
    scatterplot
=======
# Scatter plot where x is fertilizer_quantity, y is output_quantity, and the area of each dot should be a third relevant variable you find with exploration of the data.


# Calling of the methods
########################
object_1 = Agros()
object_1.data_setup()
object_1.correlate__quantities()
>>>>>>> Stashed changes
