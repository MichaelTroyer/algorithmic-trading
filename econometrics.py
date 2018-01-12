# -*- coding: utf-8 -*-
"""
Created on Tue Jan 09 21:15:19 2018

@author: michael
"""

import pandas as pd
from collections import OrderedDict
from fredapi import Fred


start_date = '2017-01-01'
fred = Fred(api_key='0ba41d84c2fee356667bfe102a1ecd6d')


def get_interest_rates(start_date, plot=True):

    rates = {
        'fed_funds_rate': fred.get_series('FEDFUNDS', observation_start=start_date),
        'four_week_tbill': fred.get_series('DTB4WK', observation_start=start_date),
        'three_month_tbill': fred.get_series('DTB3', observation_start=start_date),
        'six_month_tbill': fred.get_series('DTB6', observation_start=start_date),
        'one_year_tbill': fred.get_series('DTB1YR', observation_start=start_date),
        }

    interest_rates = pd.DataFrame()
    for name, data in rates.items():
        interest_rates[name] = data

    if plot:
        interest_rates.plot()

    return interest_rates


def get_money_supply(start_date, plot=True):
    #M1, M2
    pass


def get_inflation_rate(start_date, plot=True):
    # GDP, CPI, PPI, Retail sales
    gdp = fred.get_series('GDP', observation_start=start_date)
    pass


def get_jobless_claims(start_date, plot=True):
    # Unemployment and new jobs created
    # Average weekly hours worked and average earnings
    #Employment cost index
    pass


def get_consumer_confidence(start_date, plot=True):
    # CCI
    pass


def get_business_activity(start_date, plot=True):
    # NAPMR, durable goods orders, housing starts and building permits
    # Regional manufacturing surveys
    pass


#release = fred.search_by_category(22)
#print release


get_interest_rates(start_date)