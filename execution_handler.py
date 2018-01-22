#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 12:12:48 2018

@author: michael
"""


import shelve
import requests
import Robinhood


class ExchangeHandler():
    def __init__(self, username_email, password):
        self.trader = Robinhood.Robinhood()
        self.trader.login(username_email, password)
        self.portfolio = self.trader.portfolios()
        self.positions = self.trader.positions()['results']
        self.open_positions = self.trader.securities_owned()['results']
        self.total_value = float(self.portfolio['equity'])
        self.cash = float(self.portfolio['withdrawable_amount'])
        self.stocks = {open_position['instrument']: int(float(open_position['quantity']))
                       for open_position in self.open_positions}
        self.stocks = {self.get_Robinhood_asset_data(k)['symbol']: v 
                       for k, v in self.stocks.items()}

    def get_Robinhood_asset_data(self, api_url):
        return requests.get(api_url).json()
        
    def get_all_positions(self):
        return self.positions
    
    def get_Robinhood_portfolio_data(self):
        return {'total_value': self.total_value, 'cash': self.cash, 'assets': self.stocks}

    def sell_asset(self, symbol, shares, order_type='market'):
#        self.trader.
        pass
    
    def buy_asset(self, symbol, shares, order_type='market'):
        pass

    def get_transaction_data(self, symbol):
        pass


if __name__ == '__main__':
    d = shelve.open(r'/home/michael/Documents/databases/log.db')
    exchange = ExchangeHandler(d['username'], d['password'])
    print exchange.get_Robinhood_portfolio_data()
#    print exchange.get_all_positions()