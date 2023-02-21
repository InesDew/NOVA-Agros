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
def gapminder(year:int):
    scatterplot