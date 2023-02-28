"""
This module contains the Agros class, which is used to access, visualize and manipulate the
agricultural data related to total factor productivity.

The Agros class provides the following methods:
----------
data_setup(): Downloads the Agricultural total factor productivity (USDA) dataset, stored in
a csv file. If the folder to store the file does not exist, it is created.

country_list(): Returns a list of countries present in the dataset.

correlate_quantities(): Plots a heatmap of the correlations between different agricultural
output and input variables.

output_area_plot(country=None, normalize=False): Plots an area chart of the distinct "output"
columns for a selected country. If no country is provided or "World" is provided, the sum of
all distinct countries is plotted. If normalize is set to True, the output is normalized in
relative terms.

output_over_time(countries): Plots a line chart comparing the total output of each of the
chosen countries over the years.

gapminder(year): Visualizes the relation between the usage of fertilizer, animal feed and the
output for a given year.
"""
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns


class Agros:
    """
    Class Agros represents a collection of methods that allow for data manipulation and
    visualization of agricultural data.

    Attributes:
    ----------
    data: A pandas dataframe containing data of agricultural total factor productivity

    Methods:
    ----------
    init(self): Initializes Agros object with data attribute set to None.
    data_setup(self): Downloads data and stores it in a pandas dataframe.
    country_list(self): Returns a list of all unique countries present in the data.
    correlate_quantities(self): Computes the correlation matrix of several output-related
    columns and displays it using a heatmap.
    output_area_plot(self, country=None, normalize=False): Plots an area chart for the output
    columns of a country, normalized or not. When country=None or "World", the chart shows
    the sum of all countries.
    output_over_time(self, countries): Compares the output of several countries over time and
    displays a line chart.
    gapminder(self, year: int): Displays the relationship between fertilizer usage, animal feed
    usage, and output in given year. Raises a TypeError if the year is not an integer.
    """

    def __init__(self):
        self.data = None

    sns.set(style="whitegrid")

    def data_setup(self):
        """
        Sets up the necessary data for the class instance. Downloads a CSV file containing
        agricultural total factor productivity data from the OWID datasets repository on GitHub
        and saves it to a local directory that is only made once and is called 'downloads'.
        Then reads the CSV file into a Pandas DataFrame and assigns it to the `data` attribute
        of the class instance.

        Parameters:
        ----------
        None

        Returns:
        ----------
        None
        """
        absolute_path = os.path.dirname(__file__)
        relative_path = "/downloads"
        full_path = absolute_path + relative_path

        if not os.path.exists(full_path):
            os.mkdir(full_path)

        url = "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/Agricultural%20total%20factor%20productivity%20(USDA)/Agricultural%20total%20factor%20productivity%20(USDA).csv"
        response = requests.get(url)
        file_path = os.path.join(full_path, "dataset.csv")
        with open(file_path, "w") as f:
            f.write(response.text)

        dataframe = pd.read_csv(file_path)
        self.data = dataframe

    def country_list(self) -> list:
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

    def correlate_quantities(self):
        """
        Computes and visualizes the correlation between various quantities related to
        agricultural productivity using a heatmap.

        Parameters:
        ----------
        None

        Returns:
        ----------
        None
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
        f, ax = plt.subplots(figsize=(12, 10))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        sns.heatmap(corr, annot=True, mask=mask, cmap=cmap)

    def output_area_plot(self, country: str = None, normalize: bool = False):
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

        if country not in self.country_list():
            raise ValueError("ValueError: Country not in dataset.")

        if country is None or country == "World":
            country = "World"
            df = self.data.groupby(["Year"], as_index=False)["output"].sum()
        else:
            df = (
                self.data[self.data["Entity"] == country]
                .groupby(["Year"], as_index=False)["output"]
                .sum()
            )

        if normalize is True:
            df["output_normalized"] = (
                df["output"].apply(lambda x: x / df["output"].max()) * 100
            )
            plt.stackplot(df["Year"], df["output_normalized"])
        else:
            plt.stackplot(df["Year"], df["output"])

        plt.title(f"Output by Year ({country})")
        plt.xlabel("Year")
        plt.ylabel("Output")
        plt.show()

    def output_over_time(self, countries: str):
        """
        Receives a string with a country or a list of country strings
        and compares the total of the "_output_" columns of each country
        over the years.

        Parameters:
        -------
        countries : list of strings
            A list of countries included in the output per country dataframe.

        Returns:
        -------
        Line Plot:
            Shows the output for each of the chosen countries.
        """
        if isinstance(countries, (list, str)) is False:
            raise TypeError("The given argument 'countries' is not a list or string")

        if isinstance(countries, list) is False:
            countries = list(countries.split(" "))

        for element in countries:
            if element not in self.country_list():
                raise ValueError("ValueError: Country not in dataset.")

        df = pd.concat(
            [
                self.data[["Entity", "Year"]],
                self.data.filter(regex=r"\_output_", axis=1),
            ],
            axis=1,
        )
        df = df.groupby(["Entity", "Year"]).sum().reset_index()
        df["total_output"] = df.sum(axis=1)
        df = df[df["Entity"].isin(countries)]

        year = df["Year"]
        country = df["Entity"]
        sns.lineplot(data=df, x=year, y="total_output", hue=country)
        plt.xlabel("Year")
        plt.ylabel("Total output")
        plt.title("Comparing the total output of countries")
        plt.show()

    def gapminder(self, year: int):
        """
        Visualizes the relation between the usage of fertilizer, animal feed and the
        output for a given year. The variable 'animal_feed_quantity' was chosen because
        it showed the highest correlation with the 'output_quantity' variable next to
        'fertilizer_quantity'.

        Parameters
        ----------
        year : int
            The year for which the data will be visualized.

        Returns:
        -------
        Gapminder Plot:
            Shows the relation between output, animal feed and fertilizer.
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
        plt.show()
