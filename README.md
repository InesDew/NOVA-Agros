# Group 16
__Team Members__

53676 Celina Kollwitz  
55577 Helena Krumm  
55913 Ronja Suhling  
55944 Ines Dewever  


# Welcome to project Agros
Hello! We are four students from Nova SBE and we participated in a __two day hackathon__ promoted to study the agricultural output of several countries. We want to contribute to the green transition by having a more savvy taskforce, so we decided to create a python class for the challenge. The following file will guide you through the different methods of our analysis.

# Sources
For this project, we used the data from [Our World in Data](https://ourworldindata.org/). The dataset can be found [here](https://github.com/owid/owid-datasets/blob/master/datasets/Agricultural%20total%20factor%20productivity%20(USDA)/Agricultural%20total%20factor%20productivity%20(USDA).csv).
The data can be ckecked in detail [here](https://github.com/owid/owid-datasets/tree/master/datasets/Agricultural%20total%20factor%20productivity%20(USDA)).


# Day 1
After creating a  "Group_16" repository, we initialized the repository with a README.md file, a proper license, and a .gitignore for python. Ines, who created the repository gave the __Maintainer__ permissions to all of us. And we cloned the repository to our own laptops. 


We created a class __Agros()__ in the __group_16.py__ file, which contains several methods developed in sub branches. 
- [ ] One method  __download__ , downloads the data file into a downloads/ directory in the root directory of the project (main project directory). If the data file already exists, the method will not download it again. This method also reads the dataset into a pandas dataframe which is an attribute of our class Agros().

- [ ] A second method __name method__ that outputs a list of the available countries in the data set was created.
- [ ] A third method __name method__ that plots a way to correlate the "\_quantity" columns was written.
- [ ] A fourth method __name method__ is created, that plots an area chart of the *distinct* "\_output_" columns over the years (x-axis). This method receives two arguments: a country argument and a normalize argument. When the country argument is *NONE* or 'World', the sum for all *distinct* countries is plotted. If the normalize statement is True, the output in relative terms is plotted, so for each year, the output is always 100% If the chosen country does not exist, the method returns a ValueError.
- [ ] A fifth method __name method__ receives a string with a country or a list of country strings. This method compares the total of the "\_output_" columns for each of the chosen countries over the years (x-axis) and plots it, so a comparison can be made.
- [ ] A sixth method  __gapminder__ is developed and is a reference to the famous [gapminder tools](https://www.gapminder.org/tools/#$chart-type=bubbles&url=v1). It receives an argument year which must be an int. If the received argument is not an int, the method raises a TypeError. It plots a scatter plot where x is fertilizer_quantity, y is output_quantity, and the area of each dot is __***** Which variable???****__

After we created the group_16.py file, we created a showcase notebook __Showcase_group_16__, which imports the Agros class showcases all the methods we developed. It tells a story about our analysis and findings. 
- Through the gapminder plot for the most recent year 2019, we show world's agricultural production. 
- We analysed the three countries Norway, __**** countries**** !!!__ from three different continents in more detail. We used the fourth and fifth  methods to illustrate each country and we point out the main differences
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
- [ ] We created another method called **choropleth** which receives a year as input and raises and Error if year is not an integer. The method plots the **tfp** variable on a world map and uses a colorbar.
- [ ] We also made a **predictor** method that receives a list of countries (max 3) as input, plots the **tfp** in the dataset and then complement it with an **ARIMA** prediction up to 2050 where the same color for each country's actual and predicted data, but a different line style is used. If one or more countries on the list is not present in the Agricultural dataframe, it is ignored. If none is, an error message is raised, that reminds the user what countries are available. 

Finally, we cleaned up our project:
- We created a __yaml file__ with all the packages we used, which can be used to generate an environment where your code will be ran.
- We used __Sphinx__ to generate a docs directory that showcases the documentation of our code. 
- We updated the __README.md__ to tell the user how to start using our project.


# Compliance
- The whole project is __PEP8 compliant__. 
- Also, The whole project is compliant with __Static Type Checking__.


WIP
# Finishing up:
Let's begin by telling a story.

Start by an overall analysis with the gapminder plot for the most recent year and the year 1980. What can you see as the major changes regarding the variables you considered the most relevant on day 1?

Choose **three** countries in the list. Any countries will do, but at least one must be a member of the EU and one must be outside the EU.  
Use all other analysis methods (all you have developed except the **gapminder**, the **choropleth**, and the **predictor**) to analyse the evolution of each country's agricultural data. Write a small story about it.

End by showing **choropleth** for the most recent year and 2000.

Finish by analysing the output of **predictor**.

