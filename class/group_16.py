import pandas as pd
import requests
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from string import ascii_letters

## This is the python file for our class that will have several methods
class Agros:

## Innit method:
################
    def __init__(self):
        self.data = None

    sns.set(style="whitegrid")


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
    def country_list(self):
        """
        Receive a list of countries from the data
        
        Parameters
        ----------
        None
        
        Returns
        -------
        country_list : list
        """
        countries = self.data["Entity"].unique()
        return countries.tolist()


## Method 3:
############

#Plot an area chart of consumption (columns biofuel_consumption, coal_consumption, fossil_fuel_consumption, gas_consumption, 
# hydro_consumption, low_carbon_consumption, nuclear_consumption, oil_consumption, other_renewable_consumption, primary_energy_consumption, 
# renewables_consumption, solar_consumption, wind_consumption)
    def correlate_quantities(self):
        # Correlation between different variables
        corr = self.data[["output_quantity","crop_output_quantity","animal_output_quantity","fish_output_quantity","ag_land_quantity","labor_quantity",
                         "capital_quantity","machinery_quantity","livestock_quantity","fertilizer_quantity","animal_feed_quantity","cropland_quantity",
                         "pasture_quantity","irrigation_quantity"]].corr()
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(12, 10))
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, annot=True, mask=mask, cmap=cmap)


## Method 4:
############
# Plots an area chart of the distinct "_output_" columns
# The X-axis should be the Year. 
# Method should have two arguments: a country argument and a normalize argument. 
# The country argument, when receiving NONE or 'World' should plot the sum for all distinct countries. 
# The normalize argument, if True, normalizes the output in relative terms: each year, output should always be 100%. 
# The method should return a ValueError when the chosen country does not exist.

    def output_area_plot(self, country=None, normalize=False):
        """
        Returns an area chart of the distinct "_output_" columns for a selected country
        
        Parameters
        ----------
        country : string
        defines selected country
        when receiving none, or 'World', sum of all distinct countries is plotted
        returns ValueError if chosen country doesn't excist
        normalize : boolean
        if true, normalizes the output in relative terms, output is 100% each year
        
        Returns
        -------
        area chart of the distinct "_output_" columns for a selected country
        """

        # The method should return a ValueError when the chosen country does not exist.
        if country not in self.country_list():
            raise ValueError("ValueError: Country not in dataset.")

        # The country argument, when receiving NONE or 'World' should plot the sum for all distinct countries.
        if country is None or country == 'World':
            country = 'World'
            df = self.data.groupby(['Year'], as_index=False)['output'].sum()
        else:
            # Filters only rows with country
            df = self.data[self.data['Entity'] == country].groupby(['Year'], as_index=False)['output'].sum()

        if normalize is True:
            #df_norm = df.div(df.sum(axis=1), axis=0) * 100
            df['output_normalized'] = df['output'].apply(lambda x: x / df['output'].max())*100
            plt.stackplot(df["Year"], df["output_normalized"])
            #graph = df_norm.plot.area()
        else:
            plt.stackplot(df["Year"], df["output"])
            #graph = df.plot.area("Year", stacked=True)
    
        # Plots an area chart of the distinct "_output_" columns
        # The X-axis should be the Year.
        plt.title(f"Output by Year ({country})")
        plt.xlabel("Year")
        plt.ylabel("Output")

        plt.show()




        # The method should return a ValueError when the chosen country does not exist.
        if country not in self.country_list():
            raise ValueError("ValueError: Country not in dataset.")

        # The country argument, when receiving NONE or 'World' should plot the sum for all distinct countries.
        if country is None or country == 'World':
            country = 'World'
            df = self.data.groupby(['Year'], as_index=False)['output'].sum()
        else:
            # Filters only rows with country
            df = self.data[self.data['Entity'] == country].groupby(['Year'], as_index=False)['output'].sum()

        if normalize is True:
            df_norm = df.div(df.sum(axis=1), axis=0) * 100
            plt.stackplot(df_norm["Year"], df_norm["output"])
            #graph = df_norm.plot.area()
        else:
            plt.stackplot(df["Year"], df["output"])
            #graph = df.plot.area("Year", stacked=True)
    
        # Plots an area chart of the distinct "_output_" columns
        # The X-axis should be the Year.
        plt.title(f"Output by Year ({country})")
        plt.xlabel("Year")
        plt.ylabel("Output")

        plt.show()
## Method 5:
############

# Input: a country or a list of countries (in string format)

# Compare the total of the "_output_" columns for each of the chosen countries

# Plot it
# The X-axis should be the Year

    def output_over_time(self, countries):
        """
        Receive a string with a country or a list of country strings 
        and compare the total of the "\_output_" columns of each country 
        over the years.
        """
        
        """
        Parameters
        ----------
        countries : list of strings
            A list of countries included in the output per country dataframe
            
        Returns
        -------
        Line Plot: 
            Shows the output for each of the chosen countries.
            
        """
        df = pd.concat([self.data[["Entity","Year"]], self.data.filter(regex="\_output_", axis=1)], axis=1)
        df = df.groupby(["Entity","Year"]).sum().reset_index()
        df["total_output"] = df.sum(axis=1)
        
        if isinstance(countries, list) is False:
            countries = list(countries.split(" "))
        df = df[df["Entity"].isin(countries)]
        
        year = df["Year"]
        country = df["Entity"]
        sns.lineplot(data=df, x=year, y="total_output", hue=country)
        plt.xlabel("Year")
        plt.ylabel("Total output")
        plt.title("Comparing the total output of countries")
        plt.show()


## Method 6:
############

    def gapminder(self, year: int):
        """
        Visualizes the relation between the usage of fertilizer, animal feed and the output for a
        given year. The variable 'animal_feed_quantity' was chosen because it showed the highest
        correlation with the 'output_quantity' variable next to 'fertilizer_quantity'.

        Parameters
        ----------
        year : int
            The year for which the data will be visualized.

        Raises
        ------
        TypeError
            If the argument 'year' is not an integer.

        Example
        -------
        >>> gapminder(1990)
        """

        if isinstance(year, int) is False:
            raise TypeError("The given argument 'year' is not int.")

        fertilizer = self.data[self.data["Year"] == year]["fertilizer_quantity"]
        output = self.data[self.data["Year"] == year]["output_quantity"]
        area = self.data[self.data["Year"] == year]["animal_feed_quantity"]

        sns.scatterplot(x=fertilizer, y=output, size=area, sizes=(1, 300))
        plt.title("Understanding the relation of usage of fertilizer, animal feed and the output")
        plt.xlabel("Fertilizer Quantity")
        plt.ylabel("Output Quantity")
        plt.legend(title="Animal Feed", loc="lower right")
        plt.show()
