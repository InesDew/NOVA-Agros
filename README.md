# Group 16
__Team Members__

53676 Celina Kollwitz  
53676@novasbe.pt  
55577 Helena Krumm  
55577@novasbe.pt  
55913 Ronja Suhling  
55913@novasbe.pt  
55944 Ines Dewever  
55913@novasbe.pt  


# Welcome to project Agros
Hello! We are four students from Nova SBE and we participated in a __two day hackathon__ promoted to study the agricultural output of several countries. We want to contribute to the green transition by having a more savvy taskforce, so we decided to create a python class Agros () for the challenge. The class has 8 different methods for performing various agricultural analyses. The class is defined in the file group_16.py.

Additionally, the repository contains a Jupyter Notebook Showcase_Group_16.ipynb which showcases the usage of the Agros() class.


# Sources
For this project, we used the data from [Our World in Data](https://ourworldindata.org/). The dataset can be found [here](https://github.com/owid/owid-datasets/blob/master/datasets/Agricultural%20total%20factor%20productivity%20(USDA)/Agricultural%20total%20factor%20productivity%20(USDA).csv).
The data can be ckecked in detail [here](https://github.com/owid/owid-datasets/tree/master/datasets/Agricultural%20total%20factor%20productivity%20(USDA)).

# How to start using our project

To start using this project, you will need to follow these steps:

1) Clone this repository to your local machine using Git or download the zip file and extract it.
2) Create the ProjectAgrosEnvironment in your conda through the environment.yml file, which installs all the required packages listed in the requirements section.
3) Import the Agros() class from the group_16 module into your Python script or Jupyter Notebook or start using the Showcase_Group_16 notebook to see our analysis.


# Compliance
- The whole project is __PEP8 compliant__. 
- Also, The whole project is compliant with __Static Type Checking__.

# Specification of Requirements
Software tools on which our code finds grounding and are essential for it to run
- os
- pandas
- requests
- matplotlib.pyplot
- numpy
- seaborn
- typing
- statsmodels.tsa.arima.model
- geopandas
- pmdarima.arima
- warnings

# Day 1
After creating a  "Group_16" repository, we initialized the repository with a README.md file, a proper license, and a .gitignore for python. Ines, who created the repository gave the __Maintainer__ permissions to all of us. And we cloned the repository to our own laptops. 

We created a class __Agros()__ in the __group_16.py__ file, which contains several methods developed in sub branches. 
- [ ] 1. Method:   __data_setup__ , downloads the data file into a "downloads/" directory in the root directory of the project (main project directory). If the data file already exists, the method will not download it again. This method also reads the dataset into a pandas dataframe which is an attribute of our class Agros().
- [ ] 2. Method: __country_list__, outputs a list of the available countries in the data set was created.
- [ ] 3. Method: __correlate_quantities__, plots a way to correlate the "\_quantity" columns.
- [ ] 4. Method: __output_area_plot__, plots an area chart of the distinct "\_output_" columns over the years (x-axis). This method receives two arguments: a country argument and a normalize argument. When the country argument is *NONE* or 'World', the sum for all *distinct* countries is plotted. If the normalize statement is True, the output in relative terms is plotted, so for each year, the output is always 100% If the chosen country does not exist, the method returns a ValueError.
- [ ] 5. Method: __output_over_time__, receives a string with a country or a list of country strings. This method compares the total of the "\_output_" columns for each of the chosen countries over the years (x-axis) and plots it, so a comparison can be made.
- [ ] 6. Method:__gapminder__, is a reference to the famous [gapminder tools](https://www.gapminder.org/tools/#$chart-type=bubbles&url=v1). It receives an argument year which must be an int. If the received argument is not an int, the method raises a TypeError. It plots a scatter plot where x is fertilizer_quantity, y is output_quantity, and the area of each dot is animal_feed_quantity.

After we created the group_16.py file, we created a showcase notebook __Showcase_group_16__, which imports the Agros class and showcases all the methods we developed. It tells a story about our analysis and findings. 
- Through the gapminder plot for the most recent year 2019, we show world's agricultural production. 
- We analysed the three countries __Norway, Brazil and India__ from three different continents in more detail. We used the fourth and fifth  methods to illustrate each country and we point out the main differences
- Then, we use the third method to show how the variables correlate with each other.

We wrap up the first day with a very short overall analysis between quantities and outputs.

# Feedback Day 1
After handing in our work of Day 1, we received feedback where the following things still weren't correct:
- Gapminder should be in log-log scale or have an option for that. A log-log scale allows a better analysis over the magnitudes of the data.
- Given that all "_quantity" variables of the correlation plot are highly correlated (>0.5) you should note and discuss the ones with a relatively lower correlation like "fish_output_quantity" and "pasture_quantity", which are the ones that stand out.
- While the area plot allows for "World", it doesn't allow None.

After implementing the feedback from Day 1 and adjusting the methods, we also changed a few things that were stated as an extended scope of the Day 1 tasks. 
- [ ] First, we removed aggregated columns from our data, as we didn't notice that on day one and it makes no sense to use an Asian country and then also use Asia in our plots, for example. 
- [ ] We also added a descriptive title, correct labels, and an annotation stating the source of the data.

# Day 2
- [ ] As we now will be using geodata, we altered the download data method so that it also downloads and reads a geographical dataset. The geo dataset has the polygons for as many countries as possible and we downloaded it [here](https://www.naturalearthdata.com). We then merged the agricultural data with the geodata on the countries and made a **VARIABLE** of the class called *merge_dict* which is a dictionary that renames at least one country
- [ ] 7. Method: __choropleth__,receives a year as input and raises and Error if year is not an integer. The method plots the tfp variable on a world map and uses a colorbar.
- [ ] 8. Method: __predictor__, receives a list of countries (max 3) as input, plots the tfp in the dataset and then complement it with an ARIMA prediction up to 2050 where the same color for each country's actual and predicted data, but a different line style is used. If one or more countries on the list is not present in the Agricultural dataframe, it is ignored. If none is, an error message is raised, that reminds the user what countries are available. 

Then, we cleaned up our project:
- We created a __yaml file__ with all the packages we used, which can be used to generate an environment where your code will be ran.
- We used __Sphinx__ to generate a docs directory that showcases the documentation of our code. 
- We updated the __README.md__ to tell the user how to start using our project.

Finally, we finished up our project:
- Before starting our analysis, we showed the data used for the analysis in a dataframe and showed a list of all the countries in the dataset.
- We started by overall analysis with the gapminder plot for the most recent year 2019 and the year 1980. The major changes regarding the variables we considered the most relevant on day 1 and animal_feed_quantity are explained.
- We then applied the two new methods (output_area_plot & output_over_time) to the three selected countries from day 1 (Norway, Brazil, India) to explain their agricultural evolution further and wrote a small story about it.
- After the analysis of day 1, we took a look at the correlation between the _quantity columns
- On day 2 of the project, we continued our analysis and showed the choropleth for the year 2019 and the year 2000
- Finally, we finished our project with analyzing of the output of the predictor method.
