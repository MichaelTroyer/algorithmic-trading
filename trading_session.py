# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 21:24:59 2017

@author: michael

TODO: Sector balance
TODO: Market baseline report
"""


from datetime import date, datetime
import os
import sqlite3
import traceback

import pandas as pd


class TradingSession():
    def __init__(self,
                 name,
                 cash,
                 stocks,
                 screen,
                 strategy,
                 portfolio,
                 data_handler,
                 start_date='2016-01-01',
                 pass_stocks=None,
                 look_stocks=None,
                 pass_sectors=None,
                 look_sectors=None,
                 ):

        self.name = name
        self.cash = cash
        self.stocks = stocks
        self.symbols, self.shares = zip(*self.stocks.items())
        self.start_date = start_date
        self.today = date.today()
        self.strategy = strategy
        self.portfolio = portfolio
        self.data = data_handler
        self.screen = screen()

        self.stock_value = self.portfolio.stock_value
        self.total_value = self.stock_value + self.cash

        self.look_stocks = look_stocks if look_stocks else list()
        self.pass_stocks = pass_stocks if pass_stocks else list()
        self.look_sectors = look_sectors if look_sectors else list()
        self.pass_sectors = pass_sectors if pass_sectors else list()

        print 'Getting target sectors'
        self.top_sectors = self.screen.get_top_sectors(20)
        self.top_sectors += [ls for ls in self.look_sectors if ls not in self.top_sectors]
        self.top_sectors = [ts for ts in self.top_sectors if ts not in self.pass_sectors]
        for ix, sector in enumerate(self.top_sectors, 1):
            print '{:>2}: {}'.format(ix, sector)

        print 'Getting stocklist'
        self.stocklist = []

        seen = set()
        for sector in self.top_sectors:
#            print 'Evaluating sector: {}'.format(sector)
            sector_stocks = self.screen.stock_screen(
                    Industry       = sector,
                    TradesShareMin = 1,
                    TradesShareMax = 20,
                    TradeVolMin    = None,
                    TradeVolMax    = 1000000,
                    PERatioMin     = 0,
                    ResultsPerPage = 'OneHundred',
                    SortyBy        = 'Volume',
                    SortDirection  = 'Descending')
            for sector_stock in sector_stocks:
                sector_stock.append(sector)
                if sector_stock[0] not in seen:
                    self.stocklist.append(sector_stock)
                    seen.add(sector_stock[0])
        self.stocklist = [s for s in self.stocklist if s[0] not in self.pass_stocks]

        # Create loal databases
        print 'Connecting to session database'
        self.session_db = r'{}_session.db'.format(self.name)
        if not os.path.exists(self.session_db):
            self.create_session_database()

        self.watchlist = self.get_session_events('WATCH')

        self.update_session_metrics()

        self.portfolio.summarize()

    def create_session_database(self):
        with sqlite3.connect(self.session_db) as con:
            cur = con.cursor()

            cur.execute("CREATE TABLE Portfolio("
                        "Portfolio_Date DATE PRIMARY KEY,"
                        "Stock_Value FLOAT,"
                        "Cash_Value FLOAT,"
                        "Position_Count INT,"
                        "Annual_Return FLOAT,"
                        "Annual_Volatility FLOAT"
                        ");")

            cur.execute("CREATE TABLE Events("
                        "Symbol TEXT,"
                        "Date DATE,"
                        "Event TEXT);")

    def update_session_metrics(self):
        with sqlite3.connect(self.session_db) as con:
            cur = con.cursor()
            cur.execute("INSERT OR REPLACE INTO Portfolio "
                        "VALUES (?,?,?,?,?,?);",
                        (date.today(),
                         self.stock_value,
                         self.cash,
                         len(self.symbols),
                         self.portfolio.get_portfolio_return(),
                         self.portfolio.get_portfolio_volatility()))

    def create_session_event(self, symbol, event):
        with sqlite3.connect(self.session_db) as con:
            cur = con.cursor()
            cur.execute("INSERT INTO Events (Symbol, Date, Event) VALUES(?,?,?);",
                        (symbol, datetime.now(), event))

    def get_session_events(self, event, dedup=True):
        with sqlite3.connect(self.session_db) as con:
            df = pd.read_sql_query("SELECT * FROM Events WHERE Event = ?;",
                                   con, index_col='Symbol', parse_dates=['Date'], params=(event,))
        if dedup:
            df = df[~df.index.duplicated(keep='last')]
        return df

    def delete_session_event(self, symbol):
        with sqlite3.connect(self.session_db) as con:
            cur = con.cursor()
            cur.execute("DELETE FROM Events WHERE Symbol = ?;", (symbol,))

    def get_session_tearsheet(self):
        pass

    def evaluate_holdings(self):
        print 'Number of positions: {}'.format(len(self.symbols))
        for symbol in self.symbols:
            try:
                signal = self.strategy.process_symbol(symbol)
                if signal in ['FAIL', 'SELL']:
                    self.create_session_event(symbol, 'SELL')
            except Exception as e:
                raise Exception, '[+] Evaluate Holdings Error - {}'.format(e)

    def evaluate_stocklist(self, verbose=True):
        if verbose:
            print 'Number of stocks on stocklist: {}\n'.format(len(self.stocklist))
            print    ('| Symbol '
                      '| Price  '
                      '|   Change  '
                      '| Percent  '
                      '| Volume  '
                      '|  P/E   '
                      '| MarCap  '
                      '| Sector               '
                      '|'
                      )
            print '-' * 112
            pr_str = ('| {:<5}  | ${:<5} | ${:<8} | {:<8} | {:<7} | {:<6} | {:<7} | {:<20} |')
        for symbol in self.stocklist:
            if '.' in symbol[0]: continue
            try:
                if verbose: print pr_str.format(*symbol)
                signal = self.strategy.process_symbol(symbol[0], summarize=False, plot=False)
                # Add to the list when in sell, watch until buy
                if signal in ('SELL', 'NONE'):
                    self.create_session_event(symbol[0], 'WATCH')
            except Exception as e:
                print '[+] Evaluate Stocklist Error - {}'.format(e)

    def evaluate_watchlist(self):
        print 'Number of stocks on watchlist: {}'.format(len(self.watchlist))
        plots = True if len(self.watchlist.index.tolist()) < 15 else False
        for symbol in self.watchlist.index.tolist():
            try:
                signal = self.strategy.process_symbol(symbol, plot=plots)
                if signal == 'BUY':
                    (ret, vol, sharpe), weights = self.portfolio.test_order(symbol[0])
                    if sharpe > self.portfolio.get_portfolio_sharpe():
                        self.create_session_event(symbol, 'BUY')

            except Exception as e:
                raise Exception, '[+] Evaluate Watchlist Error - {}\n{}'.format(e, traceback.format_exc())
#                raise Exception, '[+] Evaluate Watchlist Error - {}'.format(e)


if __name__ == '__main__':
    from stock_screen import StockScreen
    from data_handler import WebToDatabase
    from portfolio import MarkowitzMeanVariance
    from strategy import MovingAverageConvergenceDivergence

    data_handle = WebToDatabase()

    start_date = '2017-07-01'

    cash = 100
    stocks = {'RVSB': 36, 'LGL': 12, 'LLNW': 1}
    symbols, shares = zip(*stocks.items())

    macd = MovingAverageConvergenceDivergence((5, 10, 10), start_date, data_handle)
    mmvp = MarkowitzMeanVariance('Test', symbols, shares, cash, start_date, data_handle)
    screen = StockScreen

    session = TradingSession(name='test',
                             cash=cash,
                             stocks=stocks,
                             screen=screen,
                             strategy=macd,
                             portfolio=mmvp,
                             data_handler=data_handle,
                             start_date=start_date)
    session.evaluate_holdings()
    session.evaluate_stocklist()
    session.evaluate_watchlist()

    print session.get_session_events('SELL')
    print session.get_session_events('WATCH')
    print session.get_session_events('BUY')
