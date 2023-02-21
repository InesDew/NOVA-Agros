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

#called gapminder

#input: a year (in integer format)
#If it is not an integer, the method should raise a TypeError

#scatter plot where x is gdp, y is total energy consumption, and the area of each dot is population.

object_1 = MyClass()
object_1.data_setup()
print(object_1.data)