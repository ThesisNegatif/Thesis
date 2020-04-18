from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from dateutil.relativedelta import relativedelta

class StockPercentChange(object):
    def add_years(self, d, years):
        """
        Args:
            d(datetime object): Date on which to add years
            years(int): Number of years to add to date d
        Returns:
            New date created by adding years to d (datetime object)
        """
        try:
            return d.replace(year = d.year + years)
        except ValueError:
            return d + (date(d.year + years, 1, 1) - date(d.year, 1, 1))

    def add_months(self, d, num_months):
        """
        Args:
            d(datetime object): Date on which to add months
            num_months(int): Number of months to add to date d
        Returns:
            New date created by adding num_months to d (datetime object)
        """
        return d + relativedelta(months=+num_months)

    def add_days(self, d, num_days):
        """
        Args:
            d(datetime object): Date on which to add days
            num_days(int): Number of days to add to date d
        Returns:
            New date created by adding num_days to d (datetime object)
        """
        return d + timedelta(days=num_days)

    def get_stock_adj_close(self, ticker, chosen_date):
        """
        Args:
            ticker(str): Ticker symbol of company on which to obtain Adj Closing Stock Prices
            chosen_date(datetime object): Date on which to obtain Adj Closing Stock Price for chosen company
        Returns:
            Adjusted Closing Stock Price of company given by ticker on chosen_date (float)
        """
        stock_df = pdr.get_data_yahoo(ticker, start = chosen_date, end = chosen_date)
        return stock_df.iloc[0]['Adj Close']

    def percent_change(self, old, new):
        """
        Args:
            old(float): Original numeric value
            new(float): New numeric value
        Returns:
            Percent change between new and old (float)
        """
        return ((new-old)/old)*100
