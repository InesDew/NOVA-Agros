# Group 16
Team Members

53676 Celina Kollwitz  
55577 Helena Krumm  
55913 Ronja Suhling  
55944 Ines Dewever  


# Welcome to project Agros
Hello! We are four students from Nova SBE and we participated in a two day hackathon promoted to study the agricultural output of several countries. We want to contribute to the green transition by having a more savvy taskforce, so we decided to create a python class for the challenge. The following file will guide you through the different methods of our analysis.





# Sources
For this project, we used the data from [Our World in Data](https://ourworldindata.org/). The dataset can be found [here](https://github.com/owid/owid-datasets/blob/master/datasets/Agricultural%20total%20factor%20productivity%20(USDA)/Agricultural%20total%20factor%20productivity%20(USDA).csv).
The data can be ckecked in detail [here](https://github.com/owid/owid-datasets/tree/master/datasets/Agricultural%20total%20factor%20productivity%20(USDA)).


# Day 1
After creating a  "Group_16", we initialized the repository with a README.md file, a proper license, and a .gitignore for python. Ines, who created the repository gave the __Maintainer__ permissions to all of us. And we cloned the repository to our own laptops. 


We created a class Agros() which contains several methods, developed in sub branches. 
- [ ] One method  _download_ , downloads the data file into a __downloads/__ directory in the root directory of the project (main project directory). If the data file already exists, the method will not download it again. This method also reads the dataset into a pandas dataframe which is an attribute of our class Agros().

- [ ] A second method that outputs a list of the available countries in the data set was created.
- [ ] A third method that plots a way to correlate the "\_quantity" columns was written.
- [ ] A fourth method is created, that plots an area chart of the *distinct* "\_output_" columns over the years (x-axis). This method receives two arguments: a __country__ argument and a __normalize__ argument. When the country argument is *NONE* or 'World', the sum for all *distinct* countries is plotted. If the normalize statement is True, the output in relative terms is plotted, so for each year, the output is always 100% If the chosen country does not exist, the method returns a ValueError.
- [ ] A fifth method receives a string with a country or a list of country strings. This method compares the total of the "\_output_" columns for each of the chosen countries over the years (x-axis) and plots it, so a comparison can be made.
- [ ] A sixth method  __gapminder__ is a method developed and is is a reference to the famous [gapminder tools](https://www.gapminder.org/tools/#$chart-type=bubbles&url=v1). It receives an argument __year__ which must be an __int__. If the received argument is not an int, the method raises a TypeError. It plots a scatter plot where __x__ is __fertilizer_quantity__, y is __output_quantity__, and the area of each dot is ***** Which variable???****

After we created the group_16.py file, we created a showcase notebook "Showcase_group_16", which imports the Agros class showcases all the methods we developed. It tells a story about our analysis and findings. 
- Through the gapminder plot for the most recent year 2019, we show world's agricultural production. 
- We analysed the three countries Norway, **** countries**** !!! from three different continents in more detail. We used the fourth and fifth  methods to illustrate each country and we point out the main differences

- Then, we use the third method to show how the variables correlate with each other.

We wrap up the first day with a very short overall analysis between quantities and outputs.


# Feedback Day 1


TBD

After implementing the feedback from Day 1, we also changed a few things that were stated as an extended scope of the Day 1 tasks. 

- [ ] First, we removed aggregated columns from our data, as we didn't notice that on day one and it makes no sense to use an Asian country and then also use *Asia* in our plots, for example. 
- [ ] We also added a descriptive title, correct labels, and an annotation stating the source of the data.


# Day 2
- [ ] As we now will be using geodata, we altered the download data method so that it also downloads and reads a geographical dataset. The geo dataset has the polygons for as many countries as possible and we downloaded it [here](https://www.naturalearthdata.com). We then merged the agricultural data with the geodata on the countries and made a **VARIABLE** of the class called *merge_dict* which is a dictionary that renames at least one country
- [ ] We created another method called **choropleth** which receives a year as input and raises and Error if year is not an integer. The method plots the **tfp** variable on a world map and uses a colorbar.
- [ ] We also made a **predictor** method that receives a list of countries (max 3) as input, plots the **tfp** in the dataset and then complement it with an **ARIMA** prediction up to 2050 where the same color for each country's actual and predicted data, but a different line style is used. If one or more countries on the list is not present in the Agricultural dataframe, it is ignored. If none is, an error message is raised, that reminds the user what countries are available. 

Finally, we cleaned up our project:
- We created a yaml file with all the packages we used, which can be used to generate an environment where your code will be ran.
- We used Sphinx to generate a __docs__ directory that showcases the documentation of our code. 
- We updated the README.md to tell the user how to start using our project.


# Compliance
- The whole project is __PEP8 compliant. 
- Also, The whole project is compliant with __Static Type Checking__.


WIP



# Finishing up:
Let's begin by telling a story.

Start by an overall analysis with the gapminder plot for the most recent year and the year 1980. What can you see as the major changes regarding the variables you considered the most relevant on day 1?

Choose **three** countries in the list. Any countries will do, but at least one must be a member of the EU and one must be outside the EU.  
Use all other analysis methods (all you have developed except the **gapminder**, the **choropleth**, and the **predictor**) to analyse the evolution of each country's agricultural data. Write a small story about it.

End by showing **choropleth** for the most recent year and 2000.

Finish by analysing the output of **predictor**.

