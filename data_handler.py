# -*- coding: utf-8 -*-
"""
Created on Sun Nov 05 08:39:17 2017

@author: michael

data_handler:
    Module for specifying classes for collecting, cleaning,
    storing, and delivering data to the strategy class.

"""


from datetime import date, datetime, timedelta
from pandas_datareader import DataReader

import os
import platform
import sqlite3
import traceback

import quandl as qd
import pandas as pd
import holidays

API_KEY = 'Njj1rcHsiQnXzaazaivH'
qd.ApiConfig.api_key = API_KEY

us_holidays = holidays.UnitedStates()


class WebToDatabase():

    def __init__(self, db_path=None):
        if not db_path:
            if platform.system() == 'Windows': root = 'C:\Users'
            if platform.system() == 'Linux': root = '/home'
            db_path = os.path.join(root, 'michael', 'Documents', 'databases', 'securities_master.db')

        # Instantiate database
        self.db_path = db_path
        if not os.path.exists(db_path):
            self.build_database(db_path)

        # Get a list of symbols and last symbol date from db
        with sqlite3.connect(self.db_path) as con:
            cur = con.cursor()
            rows = cur.execute(
                "SELECT P.Symbol, P.Date "
                "FROM Prices P "
                "INNER JOIN ( "
                    "SELECT Symbol, max(Date) as MaxDate "
                    "FROM Prices "
                    "GROUP BY Symbol"
                    ") MX "
                    "ON P.Symbol = MX.Symbol "
                    "AND "
                    "P.Date = MX.MaxDate;")
            self.symbols = dict(rows)

    def build_database(self, db_path):
        with sqlite3.connect(db_path) as con:
            cur = con.cursor()

            cur.execute("CREATE TABLE Prices("
                        "Symbol TEXT,"
                        "Date DATE,"
                        "Open FLOAT,"
                        "High FLOAT,"
                        "Low FLOAT,"
                        "Close FLOAT,"
                        "Volume INT,"
                        "Source TEXT"
                        ");")

    def get_web_data(self, symbol, src, start_date, end_date=None):
        try:
            stk_df = DataReader(symbol, src, start=start_date, end=end_date)
            stk_df.rename(columns={'Adj. Close': 'Close'}, inplace=True)
            stk_df.drop(['Adj Close'], axis=1, inplace=True)
            stk_df['Source'] = src
            stk_df['Symbol'] = symbol
            with sqlite3.connect(self.db_path) as con:
                stk_df.to_sql('Prices', con, if_exists='append')
            return stk_df
        except Exception as e:
            raise Exception, '[+] get_web_data Error - {}'.format(e)

    def get_Quandl_data(self, symbol, start_date, end_date=None):
        try:
            qd_df = qd.get('WIKI/{}'.format(symbol), start_date=start_date, end_date=end_date)
            drop_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Ex-Dividend', 'Split Ratio']
            qd_df.drop(drop_cols, axis=1, inplace=True)
            qd_df.rename(columns={'Adj. Open': 'Open',
                                  'Adj. High': 'High',
                                  'Adj. Low': 'Low',
                                  'Adj. Close': 'Close',
                                  'Adj. Volume': 'Volume'},
                                  inplace=True)
            qd_df['Source'] = 'quandl'
            qd_df['Symbol'] = symbol
            with sqlite3.connect(self.db_path) as con:
                qd_df.to_sql('Prices', con, if_exists='append')
            return qd_df
        except Exception as e:
            raise Exception, '[+] get_Quandl_data Error - {}'.format(e)

    def get_prices(self, symbol, start_date):
        try:
            with sqlite3.connect(self.db_path) as con:
                stk_df = pd.read_sql_query(
                    "SELECT * FROM Prices WHERE Symbol = ? AND Date > ?",
                    con=con, parse_dates=['Date'], params=(symbol, start_date,),
                    )
            # Unnecessary, god willing
            stk_df.drop_duplicates(['Date', 'Symbol'], inplace=True)
            stk_df.set_index('Date', inplace=True)
            stk_df.sort_index(inplace=True)
            return stk_df
        except Exception as e:
            raise Exception, '[+] get_prices Error - {}'.format(e)

    def get_DataFrame(self, symbol, start_date, verbose=False):
        try:

            # Rewind end date to last actual trading date
            last_trade_date = date.today()
            if last_trade_date in us_holidays:
                last_trade_date -= timedelta(1)
            last_day_of_week = last_trade_date.isoweekday()
            if last_day_of_week == 6:
                last_trade_date -= timedelta(1)
            if last_day_of_week == 7:
                last_trade_date -= timedelta(2)

            if symbol in self.symbols.keys():
                if verbose: print 'Symbol [{}] exists in database'.format(symbol)
                end_db_date = str(self.symbols[symbol]).split()[0]
                end_db_date = datetime.strptime(end_db_date, '%Y-%m-%d')
                if end_db_date.date() < last_trade_date:
                    # We're missing some recent data
                    # Start with day after last db date
                    if verbose: print 'Updating [{}] symbol ending dates'.format(symbol)
                    update_date = end_db_date + timedelta(1)
                    try:
                        self.get_Quandl_data(symbol, update_date)
                    except Exception as e:
                        try:
                            self.get_web_data(symbol, 'yahoo', update_date)
                        except Exception as e:
                            try:
                                self.get_web_data(symbol, 'google', update_date)
                            except Exception as e:
                                raise Exception, 'Could not locate end data for [{}]\n{}'.format(e, traceback.format_exc())
                    # Update self.symbols dict
                    self.symbols[symbol] = last_trade_date

            else:
                if verbose: print 'Adding new symbol data [{}]'.format(symbol)
                try:
                    self.get_Quandl_data(symbol, None)
                except Exception as e:
                    try:
                        self.get_web_data(symbol, 'yahoo', None)
                    except Exception as e:
                        try:
                            self.get_web_data(symbol, 'google', None)
                        except Exception as e:
                            raise Exception, 'Could not locate new data for [{}]\n{}'.format(e, traceback.format_exc())
                self.symbols[symbol] = last_trade_date
            return self.get_prices(symbol, start_date)
        except Exception as e:
            if verbose:
                raise Exception, '[+] get_DataFrame Error - {}\n{}'.format(e, traceback.format_exc())
            else:
                raise Exception, '[+] get_DataFrame Error - {}'.format(e)

if __name__ == '__main__':

    # Testing
    data = WebToDatabase()
    chs = data.get_DataFrame('CHS', '2011-02-01')
    rvsb = data.get_DataFrame('RVSB', '2014-01-01')
    print chs.head()
    print chs.tail()
    print rvsb.head()
    print rvsb.tail()