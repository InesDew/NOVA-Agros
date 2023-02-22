import pandas as pd
import requests
import os

## This is the python file for our class that will have several methods
class MyClass:

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
        url = "https://nyc3.digitaloceanspaces.com/owid-public/data/energy/owid-energy-data.csv"
        #4. We download the info from this url using the requests library and put the response.text in a csv file dataset.csv that we create in the downloads dir
        response = requests.get(url)
        file_path = os.path.join(full_path,'dataset.csv')
        with open(file_path, "w") as f:
            f.write(response.text)

        #5. Read the dataset.csv file into a pandas dataframe and make it an attribute of our class (self.data)
        #6. Only take into account data after 1970 (1970 included)
        dataframe = pd.read_csv(file_path)
        dataframe = dataframe[dataframe["year"]>=1970]
        self.data = dataframe
        

        



## Method 2:
############

# Show a list of all available countries in the dataset


## Method 3:
############

#Plot an area chart of consumption (columns biofuel_consumption, coal_consumption, fossil_fuel_consumption, gas_consumption, 
# hydro_consumption, low_carbon_consumption, nuclear_consumption, oil_consumption, other_renewable_consumption, primary_energy_consumption, 
# renewables_consumption, solar_consumption, wind_consumption)

#needs a country argument and a normalize argument

#return a ValueError when the chosen country does not exist


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
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")


    def gapminder(year: int):
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
        if isinstance(year) != int:
            raise TypeError("The given argument 'year' is not int.")

        fertilizer = dataframe[dataframe["Year"] == year]["fertilizer_quantity"]
        output = dataframe[dataframe["Year"] == year]["output_quantity"]
        area = dataframe[dataframe["Year"] == year]["animal_feed_quantity"]

        sns.scatterplot(x=fertilizer, y=output, size=area, sizes=(1, 300))
        plt.title(
            "Understanding the relation of usage of fertilizer, animal feed and the output"
        )
        plt.xlabel("Fertilizer Quantity")
        plt.ylabel("Output Quantity")
        plt.legend(title="Animal Feed", loc="lower right")
        plt.show()

object_1 = MyClass()
object_1.data_setup()
print(object_1.data)
