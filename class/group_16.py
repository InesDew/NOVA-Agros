"""
The Agros module provides a class that can be used to download,
load, and analyze agricultural productivity data from the OWID
repository.

Dependencies:
-------------
    - os
    - pandas
    - requests
    - matplotlib.pyplot
    - numpy
    - seaborn
    - typing

Classes:
--------
    Agros

Methods:
--------
    data_setup() -> None:
        Downloads the agricultural productivity dataset from the OWID repository
        and stores it in a local directory called 'downloads'. If the directory
        doesn't exist, it creates one. Then, it loads the dataset into a pandas
        DataFrame and assigns it to the 'data' attribute of the object.

    country_list() -> List[str]:
        Returns a list of unique country names present in the dataset.

    correlate_quantities() -> None:
        Computes the correlation matrix between various agricultural quantities
        using data in the data attribute of the object. It then creates a heatmap
        plot of the correlation matrix using seaborn.

    output_area_plot(country: str = None, normalize: bool = False) -> None:
        Creates a stacked area plot of crop, animal, and fish output for a given
        country or the world over time. If the normalize parameter is set to True,
        the plot will show the proportion of each output type instead of the raw output values.

    output_over_time(countries: Union[str, List[str]]) -> None:
        Displays a line plot of the total output over time for one or more countries.

    def gapminder(self, year: int) -> None:
        Create a scatter plot to visualize the relationship between the usage of fertilizer,
        animal feed, and the agricultural output for a given year.

Raises:
-------
    TypeError: If the country parameter of output_area_plot() is not a string or if the
    normalize parameter is not a boolean.
    ValueError: If the country parameter of output_area_plot() is not in the dataset.
    TypeError: If the argument countries of output_over_time() is not a list or a string,
    or if the elements in the list are not strings.
    ValueError: If any of the country names in argument countries of output_over_time() is
    not present in the dataset.
    TypeError: if the year argument of gapminder() is not an integer.

Examples:
---------
    # Create an instance of Agros
    agros = Agros()

    # Download and load the agricultural productivity dataset from the OWID repository
    agros.data_setup()

    # Display a list of unique country names present in the dataset
    countries = agros.country_list()
    print(countries)

    # Compute the correlation matrix and plot it as a heatmap
    agros.correlate_quantities()

    # Create a stacked area plot of crop, animal, and fish output for a given country
    or the world over time
    agros.output_area_plot(country='United States', normalize=True)

    # Display a line plot of the total output over time for one or more countries
    agros.output_over_time(countries=['India', 'China'])

    # Display a scatter plot to visualize the relationship between the usage of fertilizer,
    animal feed, and the agricultural output for 1990.
    agros.gapminder(1990)
"""

import os
from typing import List, Union
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import geopandas as gpd
import zipfile
import urllib


class Agros:
    """
    The Agros class provides methods to download and load agricultural productivity
    data from the OWID repository, compute a correlation matrix and a stacked area
    plot, and display a line plot of the total output over time for one or more countries.

    Attributes:
    -----------
    data: a pandas DataFrame object that stores the agricultural productivity dataset.

    Methods:
    --------
    data_setup(): Downloads the agricultural productivity dataset from the OWID repository
    and stores it in a local directory called 'downloads'. If the directory doesn't exist,
    it creates one. Then, it loads the dataset into a pandas DataFrame and assigns it to
    the 'data' attribute of the object.
    country_list(): Returns a list of unique country names present in the dataset.
    correlate_quantities(): Computes the correlation matrix between various agricultural
    quantities using data in the data attribute of the object. It then creates a heatmap
    plot of the correlation matrix using seaborn.
    output_area_plot(country: str = None, normalize: bool = False): Creates a stacked area
    plot of crop, animal, and fish output for a given country or the world over time. If the
    normalize parameter is set to True, the plot will show the proportion of each output type
    instead of the raw output values.
    output_over_time(countries: Union[str, List[str]]): Displays a line plot of the total output
    over time for one or more countries.

    Raises:
    -------
    TypeError: if the country parameter is not a string or if the normalize parameter is
    not a boolean.
    ValueError: if the country parameter is not in the dataset.
    """

    def __init__(self):
        self.data = None

    def data_setup(self) -> None:
        """
        Downloads the agricultural productivity dataset from the OWID repository and stores it
        in a local directory called 'downloads'. If the directory doesn't exist, it creates one.
        Then, it loads the dataset into a pandas DataFrame and assigns it to the 'data' attribute
        of the object.

        Raises:
        -------
        None.

        Returns:
        --------
        None.
        """
        absolute_path = os.path.dirname(__file__)
        relative_path = "/downloads"
        full_path = absolute_path + relative_path

        if not os.path.exists(full_path):
            os.mkdir(full_path)

        # agricultural dataset
        url = "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/Agricultural%20total%20factor%20productivity%20(USDA)/Agricultural%20total%20factor%20productivity%20(USDA).csv"
        response = requests.get(url)
        file_path = os.path.join(full_path, "dataset.csv")
        with open(file_path, "w") as file:
            file.write(response.text)

        dataframe = pd.read_csv(file_path)
        aggregated_columns = ['Asia', 'Central Asia', 'Developed Asia', 'Northeast Asia', 'South Asia', 'Southeast Asia', 
                              'West Asia', 'Central Europe', 'Europe', 'Northern Europe', 'Southern Europe', 'Western Europe', 
                              'Central Africa', 'East Africa', 'Horn of Africa', 'North Africa', 'Southern Africa', 
                              'Sub-Saharan Africa', 'West Africa', 'Oceania', 'Central America', 
                              'Latin America and the Caribbean', 'North America', 'Developed countries', 
                              'Least developed countries', 'Sahel', 'Caribbean', 'Eastern Europe', 'Pacific', 
                              'High income', 'Low income', 'Lower-middle income', 'Upper-middle income', 'World']
        dataframe = dataframe[~dataframe['Entity'].isin(aggregated_columns)]

        # geographical dataset
        geo_dataframe = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

        # rename some Entity names in the pandas dataframe
        self.merg_dict = {'Bosnia and Herzegovina' : 'Bosnia and Herz.', 'Central African Republic': 'Central African Rep.', 
                          'Czechoslovakia': 'Czechia', 'Democratic Republic of Congo' : 'Dem. Rep. Congo', 
                          'Dominican Republic' : 'Dominican Rep.', 'Equatorial Guinea' : 'Eq. Guinea', 'Eswatini' : 'eSwatini', 
                          'Serbia and Montenegro' : 'Serbia', 'Solomon Islands' : 'Solomon Is.', 'South Sudan' : 'S. Sudan',
                          'Timor' : 'Timor-Leste', 'United States' : 'United States of America'}
        dataframe=dataframe.replace({"Entity": self.merg_dict})

        # merge the 2 dataframes
        merged_dataframe = geo_dataframe.merge(dataframe, how='right', left_on='name', right_on='Entity')

        """
        No translation/equivalent for these Entities from the pandas_dataframe: Bahrain, Cape Verde, Comoros, Former Soviet Union, French Guiana, Malta
        Mauritius, Micronesia, Polynesia, Sao Tome and Principe, Yugoslavia
        """

        self.data = merged_dataframe

    def country_list(self) -> list:
        """
        The method country_list returns a list of unique country names present in the dataset.

        Returns:
        --------
        A list of unique country names present in the dataset.

        Raises:
        -------
        None.
        """
        countries = self.data["Entity"].unique()
        return countries.tolist()

    def correlate_quantities(self) -> None:
        """
        The correlate_quantities method computes the correlation matrix between various
        agricultural quantities using data in the data attribute of the object. It then creates
        a heatmap plot of the correlation matrix using seaborn.

        Returns:
        --------
        None

        Raises:
        -------
        None.
        """
        corr = self.data[
            [
                "output_quantity",
                "crop_output_quantity",
                "animal_output_quantity",
                "fish_output_quantity",
                "ag_land_quantity",
                "labor_quantity",
                "capital_quantity",
                "machinery_quantity",
                "livestock_quantity",
                "fertilizer_quantity",
                "animal_feed_quantity",
                "cropland_quantity",
                "pasture_quantity",
                "irrigation_quantity",
            ]
        ].corr()

        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(11, 9))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        ax = sns.heatmap(corr, annot=True, mask=mask, cmap=cmap)
        ax.set(title ="Correlation matrix between quantity variables")
        plt.figtext(0,-0.1, 'Source: Agricultural total factor productivity, 2022 USDA', fontsize=10, va="top", ha="left")

        plt.show()

    def output_area_plot(self, country: str = None, normalize: bool = False) -> None:
        """
        The output_area_plot method creates a stacked area plot of crop, animal, and fish
        output for a given country or the world over time. If the normalize parameter is
        set to True, the plot will show the proportion of each output type instead of the
        raw output values.

        Parameters:
        -----------
        country (str, optional): Name of the country to plot data for. If not specified or
        set to "World", the data for the entire world will be plotted. Default is None.
        normalize (bool, optional): If True, the output values will be normalized to show the
        percentage of each output type. If False, the raw output values will be shown.
        Default is False.

        Raises:
        -------
        TypeError: If country is not a string or normalize is not a boolean.
        ValueError: If country is not in the dataset.

        Returns:
        --------
        None.
        """

        if not isinstance(country, (str, type(None))):
            raise TypeError("TypeError: Argument country is not string")

        if not isinstance(normalize, bool):
            raise TypeError("TypeError: Argument normalize is not boolean")

        if country not in self.country_list():
            if country is None or country == "World":
               country = "the World"
               dataframe = self.data.groupby(["Year"], as_index=False)[[
                   "crop_output_quantity", 
                   "animal_output_quantity", 
                   "fish_output_quantity",
                   ]].apply(sum) 
            else:
                raise ValueError("ValueError: Country not in dataset.")

        else:
            dataframe = (self.data[self.data["Entity"] == country].
                         groupby(["Year"], as_index=False)[[
                             "crop_output_quantity", 
                             "animal_output_quantity",
                             "fish_output_quantity",
                             ]]).apply(sum)
        
        if normalize is True:
            dataframe_output = dataframe[
                [
                    "crop_output_quantity",
                    "animal_output_quantity",
                    "fish_output_quantity",
                ]
            ].apply(lambda x: x / x.sum() * 100, axis=1)
            
            dataframe_norm = pd.concat([dataframe["Year"], dataframe_output], axis=1)
           
            plt.stackplot(
                dataframe_norm["Year"],
                dataframe_norm["crop_output_quantity"],
                dataframe_norm["animal_output_quantity"],
                dataframe_norm["fish_output_quantity"],
            )

        else:
            plt.stackplot(
                dataframe["Year"],
                dataframe["crop_output_quantity"],
                dataframe["animal_output_quantity"],
                dataframe["fish_output_quantity"],
            )

        plt.title(f"Output by Year in {country}.")
        plt.xlabel("Year")
        plt.ylabel("Output")
        plt.legend(["Crop Output", "Animal Output", "Fish Output"], loc="upper left")
        plt.annotate('Source: Agricultural total factor productivity, 2022 USDA', (0,0), (-80,-20), fontsize=6, 
             xycoords='axes fraction', textcoords='offset points', va='top')

        plt.show()

    def output_over_time(self, countries: Union[str, List[str]]) -> None:
        """
        The output_over_time method takes a list of country names or a string of
        space-separated country names and displays a line plot of the total output
        over time for each country.

        Args:
        -----
        countries: A list of country names or a string of space-separated country names.

        Raises:
        -------
        TypeError: If the argument countries is not a list or a string, or if the elements
        in the list are not strings.
        ValueError: If any of the country names in countries is not present in the dataset.

        Returns:
        --------
        None.

        Displays a line plot of the total output over time for each country in the input list.
        """
        if not isinstance(countries, list) and not isinstance(countries, str):
            raise TypeError("The given argument 'countries' is not a list or a string")

        if isinstance(countries, list):
            for element in countries:
                if not isinstance(element, str):
                    raise TypeError(
                        "The given argument 'countries' is not a list of strings"
                    )

        if isinstance(countries, list) is False:
            countries = list(countries.split(" "))

        for element in countries:
            if element not in self.country_list():
                raise ValueError("ValueError: Country not in dataset.")

        dataframe = pd.concat(
            [
                self.data[["Entity", "Year"]],
                self.data.filter(regex=r"\_output_", axis=1),
            ],
            axis=1,
        )
        dataframe = dataframe.groupby(["Entity", "Year"]).sum().reset_index()
        dataframe["total_output"] = dataframe.sum(axis=1, numeric_only=True)
        dataframe = dataframe[dataframe["Entity"].isin(countries)]

        year = dataframe["Year"]
        country = dataframe["Entity"]
        sns.lineplot(data=dataframe, x=year, y="total_output", hue=country)
        plt.xlabel("Year")
        plt.ylabel("Total output")
        plt.title("Comparing the total output of countries")
        plt.annotate('Source: Agricultural total factor productivity, 2022 USDA', (0,0), (-80,-20), fontsize=6, 
             xycoords='axes fraction', textcoords='offset points', va='top')

        plt.show()

    def gapminder(self, year: int) -> None:
        """
        Create a scatter plot to visualize the relationship between the usage of fertilizer,
        animal feed, and the agricultural output for a given year.

        Args:
        -----
        year (int): the year for which to create the scatter plot.

        Raises:
        -------
        TypeError: if year is not an integer.

        Returns:
        --------
        None.

        The scatter plot will have the following:
        The x-axis represents the fertilizer quantity used for a given year.
        The y-axis represents the output quantity for a given year.
        The size of each data point represents the animal feed quantity used for a given year.
        The legend displays the range of animal feed quantities used, with larger circles
        indicating higher values.

        The title, x-axis label, and y-axis label of the scatter plot are set to appropriate values.
        The scatter
        """
        if isinstance(year, int) is False:
            raise TypeError("The given argument 'year' is not int.")

        fertilizer = self.data[self.data["Year"] == year]["fertilizer_quantity"]
        output = self.data[self.data["Year"] == year]["output_quantity"]
        area = self.data[self.data["Year"] == year]["animal_feed_quantity"]

        sns.scatterplot(x=fertilizer, y=output, size=area, sizes=(1, 300))
        plt.title(
            "Understanding the relation of usage of fertilizer, animal feed and the output"
        )
        plt.xlabel("Fertilizer Quantity")
        plt.ylabel("Output Quantity")
        plt.legend(title="Animal Feed", loc="lower right")
        plt.annotate('Source: Agricultural total factor productivity, 2022 USDA', (0,0), (-80,-20), fontsize=6, 
             xycoords='axes fraction', textcoords='offset points', va='top')

        plt.show()

    def choropleth(self, year: int) -> None:

        if isinstance(year, int) is False:
            raise TypeError("The given argument 'year' is not int.")
        
        data_year = self.data[self.data["Year"] == year]
        data_year.plot(column = 'tfp', legend = True, figsize = [20,10], legend_kwds = {'label': "TFP by country"}) 
        plt.annotate('Source: Agricultural total factor productivity, 2022 USDA', (0,0), (-80,-20), fontsize=6, 
             xycoords='axes fraction', textcoords='offset points', va='top')
        plt.title(f"TFP in Year {year} per country.")

        
        """
        Make a method called choropleth. 
        1. OK This method should receive a year as input, which must be an integer. 
        2. OK Raise otherwise. 

        3. OK We're going deep with our analysis and we're going to use geodata. We recommend you install geopandas. 
        4. OK Alter the method where you download the data to also download and read a geographical dataset. 

        5. OK The geo dataset must have the polygons for as many countries as possible. 
        You can get such a datafile here (use cultural data, as it refers to countries). 
        If you download a zip file, remember you want to access the shapefile (.shp) inside. 
        Read the documentation for geopandas for examples on how to read data.

        6. OK Merge (pandas equivalent to SQL JOIN) the agricultural data with the geodata on the countries. 
        Make sure the left dataframe is the geopandas dataframe. 
        
        7. OK When you plot the result of the merge, you may notice some important country or countries missing. 
        That is because their names don't match. Make a VARIABLE of the class called merge_dict which is a dictionary 
        that renames at least one country

        8. Plot the tfp variable on a world map. Make sure you use a colorbar. This example should help.
        """
        
    def predictor(self, countries: List[str]) -> None:

        if not isinstance(countries, list):
            raise TypeError("The given argument 'countries' is not a list")
            
        if isinstance(countries, list):
            # only allowing 3 countries or less
            if len(countries) > 3:
                raise TypeError(
                    "The given argument 'countries' can only contain up to 3 countries. Please reduce the number of counries."
                )
            # checking if the counries are str
            for element in countries[:]:
                if not isinstance(element, str):
                    raise TypeError(
                        "The given argument 'countries' is not a list of strings"
                    )
                # removing counries that aren't in dataset
                if element not in self.country_list():
                    countries.remove(element)
            
            # Reminding user, which countries are available, if list is empty from beginning on 
            # or if all countries were removed because they weren't part of list
            if len(countries) == 0:
                raise TypeError(
                    "Please choose three of the following countries: "+ ", ".join(self.country_list())
                )
            
            # Year and tfp columns, year as index
    #        data = self.data[['Year', 'Entity', 'tfp']]
    #        data.set_index('Year', inplace=True)
            
    #        fig, ax = plt.subplots(figsize=(10, 6))
    #        line_styles = ['-', '--', '-.'] #new
    #        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(countries)] #new
    #        for i, country in enumerate(countries):
    #            tfp = data[data['Entity'] == country]['tfp']
    #            tfp.plot(ax=ax, label=country) # color=colors[i], linestyle=line_styles[i])
        
            # fit ARIMA model and predict
    from pmdarima import auto_arima 
    import warnings 
    warnings.filterwarnings("ignore") 
        
    for i, country in enumerate(countries):
                tfp = data[data['Entity'] == country]['tfp']
                #model = ARIMA(tfp, order=(1,1,1))
                #model_fit = model.fit()
                
                # Convert the index to a format supported by ARIMA model
 #   tfp.index = pd.DatetimeIndex(tfp.index).to_period('M')
                
    model = auto_arima(data['tfp'], start_p = 1, start_q = 1, max_p = 3,
                       max_q = 3, 
                       m = 12, 
                       start_P = 0, 
                       seasonal = False, 
                       d = None, 
                       D = 1,
                       trace = True, 
                       error_action ='ignore',   # Ignore incompatible settings
                       suppress_warnings = True,  
                       stepwise = True)  
    model.summary()         
    predictions = pd.DataFrame(model.predict(n_periods = 31))
    predictions['Year'] = pd.date_range(start='2020', periods=31, freq='YS')
    predictions.head()
    
    predictions.set_index('Year', inplace=True)
                
           #     model_fit = model.fit()
                
                # Create a date range for the prediction period
             #   pred_range = pd.period_range(2020, 2050-12, freq='M')
                # Make predictions for the date range
             #   predictions = model_fit.predict(start=pred_range[0], end=pred_range[-1], typ='levels')
    
                # Set the index of the predictions to the date range
             #   predictions.index = pred_range
   
                #predictions = model_fit.predict(start='2019', end='2050')
    tfp.plot(ax=ax, label='', color=colors[i], linestyle='--') #color=colors[i], (before linestyle)
    predictions.plot(ax=ax, label='', color=colors[i], linestyle='--') #color=colors[i] (before linestyle)
                
              
    ax.set_xlabel('Year')
    ax.set_ylabel('TFP')
    ax.set_title('Total Factor productivity by Country with ARIMA Predictions')
    plt.legend()
    plt.show() 
            

            """
            1.   OK   Make a predictor method that receives a list of countries as input, up to three. 
            2.   OK   If one or more countries on the list is not present in the Agricultural dataframe, it should be ignored. 
            3.   OK   If none is, raise an error message reminding the user what countries are available. 
            4.          The predictor method should plot the tfp in the dataset and then complement it with an ARIMA prediction up to 2050. 
            5.          Use the same color for each country's actual and predicted data, but a different line style.
            """
        
        
        
        
        