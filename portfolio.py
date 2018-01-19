#
# DX Analytics
# Mean Variance Portfolio
# portfolio.py
#
#
# DX Analytics is a financial analytics library, mainly for
# derviatives modeling and pricing by Monte Carlo simulation
#
# (c) Dr. Yves J. Hilpisch
# The Python Quants GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or any later version.


import math
import traceback
import numpy as np
import pandas as pd
import datetime as dt
import scipy.optimize as sco
import scipy.interpolate as sci
import matplotlib
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod


class Portfolio(object):
    """
    Portfolio is an abstract base class providing an interface for
    all subsequent (inherited) portfolio objects.

    Enforces:
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def summarize(self):
        """
        Returns a string of summary metrics.
        """
        raise NotImplementedError("Should implement summary()")

    @abstractmethod
    def plot(self):
        """
        Plots the portfolio.
        """
        raise NotImplementedError("Should implement plot()")

    @abstractmethod
    def size_order(self):
        """
        Calculates the optimal size of a purchase position
        """
        raise NotImplementedError("Should implement size_order()")

    @abstractmethod
    def test_order(self):
        """
        Calculates the risk/return of the portfolio after addition of a symbol.
        """
        raise NotImplementedError("Should implement size_order()")

    @abstractmethod
    def get_portfolio_return(self):
        """
        Returns the average return of the weighted portfolio
        """
        raise NotImplementedError("Should implement get_portfolio_return()")

    @abstractmethod
    def get_portfolio_variance(self):
        """
        Returns the average variance of the weighted portfolio
        """
        raise NotImplementedError("Should implement get_portfolio_variance()")

    @abstractmethod
    def get_portfolio_sharpe(self):
        """
        Returns the unadjusted Sharpe ratio of the weighted portfolio
        """
        raise NotImplementedError("Should implementget_portfolio_sharpe()")

    @abstractmethod
    def get_portfolio_volatility(self):
        """
        Returns the average volatility of the portfolio
        """
        raise NotImplementedError("Should implement get_volatility()")


class MarkowitzMeanVariance(Portfolio):
    '''
    Class to implement the mean variance portfolio theory of Markowitz
    '''

    def __init__(self, name, symbols, shares, cash, start, data_handler):

        self.name = name
        self.symbols = symbols
        self.shares = shares
        self.cash = cash
        self.start_date = start
        self.data_handler = data_handler

        self.number_of_assets = len(self.symbols)
        self.final_date = dt.date.today()

        self.weights = shares
        self.weights = np.array([float(wt) / sum(self.weights) for wt in self.weights])

        self.load_data()
        self.make_raw_stats()
        self.apply_weights()

        self.prices = np.array([self.data[symbol][-1] for symbol in self.symbols])
        self.price_weights = self.prices * self.weights
        self.price_weights /= sum(self.price_weights)
        self.set_weights(self.price_weights)

        self.stock_value = sum(self.prices * self.shares)
        self.total_value = self.stock_value + self.cash

    def summarize(self):
        pad = 63
        print '\n', 'Portfolio Summary'.center(pad, '-'), '\n'
        print 'Date: {}\n'.format(dt.date.today())
        print 'Portfolio value:           $ {:6.2f}'.format(self.stock_value)
        print 'Portfolio cash:            $ {:6.2f}'.format(self.cash)
        print 'Portfolio total value:     $ {:6.2f}'.format(self.stock_value + self.cash)
        print 'Annualized Return:     {:10.3f}'.format(self.portfolio_return)
        print 'Annualized Volatility: {:10.3f}'.format(math.sqrt(self.variance))
        print 'Sharpe ratio:          {:10.3f}'.format(self.portfolio_return / math.sqrt(self.variance))
        print '\n' + 'Positions'.center(pad, '-')
        print '| Symbol  | Shares  |  Price   |  Value   | Weight  | Ret Con |'
        print '-' * pad
        for i in range(len(self.symbols)):
            print '| {:<7} |  {:<5}  | ${:>7} | ${:>7} | {:7.3f} | {:7.3f} |' \
                ''.format(
                    self.symbols[i], self.shares[i], round(self.prices[i], 2),
                    round(self.shares[i] * self.prices[i], 2),
                    self.weights[i], self.mean_returns[i])
        print '-' * pad + '\n'
        opt_weights = self.get_optimal_weights('Sharpe')
        opt_shares = self.size_order(opt_weights, self.prices, self.total_value)
        print 'Optimization'.center(pad, '-')
        print '|  Symbol  |   Weight   |   Shares    |   Current  |  Delta   |'
        print '-' * pad
        for sym, wt, oshr, cshr in zip(self.symbols, opt_weights, opt_shares, self.shares):
            print '|   {:<4}   |   {:6.4f}   |     {:<4}    |    {:<4}    |   {:>4}   |'.format(
                    sym, wt, oshr, cshr, oshr - cshr)
        print '-' * pad + '\n'
        returns, volatility, sharpe = self.test_weights(opt_weights)
        print 'Optimal return:       {:.3}'.format(returns)
        print 'Optimal volatility:   {:.3}'.format(volatility)
        print 'Optimal Sharpe ratio: {:.3}'.format(sharpe)
        print
        self.plot()
        print

    def load_data(self):
        '''
        Loads asset values from the data handler.
        '''
        self.data = pd.DataFrame()
        for sym in self.symbols:
            try:
                self.data[sym] = self.data_handler.get_DataFrame(
                  sym, self.start_date)['Close']
            except Exception as e:
                print traceback.format_exc()
                raise IOError, 'Could not locate data for {}\n{}'.format(sym, e)
        self.data.columns = self.symbols

    def make_raw_stats(self):
        '''
        Computes returns and variances
        '''
        self.raw_returns = np.log(self.data / self.data.shift(1))
        self.mean_raw_return = self.raw_returns.mean()
        self.raw_covariance = self.raw_returns.cov()

    def apply_weights(self):
        '''
        Applies weights to the raw returns and covariances
        '''
        self.returns = self.raw_returns * self.weights
        self.mean_returns = self.returns.mean() * 252
        self.portfolio_return = np.sum(self.mean_returns)

        self.variance = np.dot(
            self.weights.T, np.dot(self.raw_covariance * 252, self.weights))

    def test_weights(self, weights):
        '''
        Returns the theoretical portfolio return, portfolio volatility
        and Sharpe ratio for given weights.
        '''
        weights = np.array(weights)
        portfolio_return = np.sum(self.raw_returns.mean() * weights) * 252
        portfolio_vol = math.sqrt(
            np.dot(weights.T, np.dot(self.raw_covariance * 252, weights)))

        return np.array([portfolio_return, portfolio_vol,
                         portfolio_return / portfolio_vol])

    def set_weights(self, weights):
        '''
        Sets new weights

        Parameters
        ==========
        weights: interable
            new set of weights
        '''
        try:
            weights = np.array(weights)
            weights_sum = sum(weights).round(3)
        except:
            msg = 'weights must be an interable of numbers'
            raise TypeError(msg)

        if weights_sum != 1:
            print weights
            raise ValueError('Sum of weights must be one')

        if len(weights) != self.number_of_assets:
            msg = 'Expected %s weights, got %s'
            raise ValueError(msg % (self.number_of_assets,
                                    len(weights)))
        self.weights = weights
        self.apply_weights()

    def get_weights(self):
        '''
        Returns a dictionary with entries symbol:weights
        '''
        d = dict()
        for i in range(len(self.symbols)):
            d[self.symbols[i]] = self.weights[i]
        return d

    def get_portfolio_return(self):
        '''
        Returns the average return of the weighted portfolio
        '''
        return self.portfolio_return

    def get_portfolio_variance(self):
        '''
        Returns the average variance of the weighted portfolio
        '''
        return self.variance

    def get_portfolio_sharpe(self):
        '''
        Returns the unadjusted Sharpe ratio of the weighted portfolio
        '''
        return self.portfolio_return / math.sqrt(self.variance)

    def get_portfolio_volatility(self):
        '''
        Returns the average volatility of the portfolio
        '''
        return math.sqrt(self.variance)

    def optimize(self, target, constraint=None, constraint_type='Exact'):
        '''
        Optimize the weights of the portfolio according to the value of the
        string 'target'

        Parameters
        ==========
        target: string
            one of:

            Sharpe: maximizes the ratio return/volatility
            Vol: minimizes the expected volatility
            Return: maximizes the expected return

        constraint: number
            only for target options 'Vol' and 'Return'.
            For target option 'Return', the function tries to optimize
            the expected return given the constraint on the volatility.
            For target option 'Vol', the optimization returns the minimum
            volatility given the constraint for the expected return.
            If constraint is None (default), the optimization is made
            without concerning the other value.

        constraint_type: string, one of 'Exact' or 'Bound'
            only relevant if constraint is not None.
            For 'Exact' (default) the value of the constraint must be hit
            (if possible), for 'Bound', constraint is only the upper/lower
            bound of the volatility or return resp.
        '''
        weights = self.get_optimal_weights(target, constraint, constraint_type)
        if weights is not False:
            self.set_weights(weights)
        else:
            raise ValueError('Optimization failed.')

    def get_capital_market_line(self, riskless_asset):
        '''
        Returns the capital market line as a lambda function and
        the coordinates of the intersection between the captal market
        line and the efficient frontier

        Parameters
        ==========

        riskless_asset: float
            the return of the riskless asset
        '''
        x, y = self.get_efficient_frontier(100)
        if len(x) == 1:
            raise ValueError('Efficient Frontier seems to be constant.')
        f_eff = sci.UnivariateSpline(x, y, s=0)
        f_eff_der = f_eff.derivative(1)

        def tangent(x, rl=riskless_asset):
            return f_eff_der(x) * x / (f_eff(x) - rl) - 1

        left_start = x[0]
        right_start = x[-1]

        left, right = self.search_sign_changing(
            left_start, right_start, tangent, right_start - left_start)
        if left == 0 and right == 0:
            raise ValueError('Can not find tangent.')

        zero_x = sco.brentq(tangent, left, right)

        opt_return = f_eff(zero_x)
        cpl = lambda x: f_eff_der(zero_x) * x + riskless_asset
        return cpl, zero_x, float(opt_return)

    def get_efficient_frontier(self, n):
        '''
        Returns the efficient frontier in form of lists containing the x and y
        coordinates of points of the frontier.

        Parameters
        ==========
        n : int >= 3
            number of points
        '''
        if type(n) is not int:
            raise TypeError('n must be an int')
        if n < 3:
            raise ValueError('n must be at least 3')

        min_vol_weights = self.get_optimal_weights('Vol')
        min_vol = self.test_weights(min_vol_weights)[1]
        min_return_weights = self.get_optimal_weights('Return', constraint=min_vol)
        min_return = self.test_weights(min_return_weights)[0]
        max_return_weights = self.get_optimal_weights('Return')
        max_return = self.test_weights(max_return_weights)[0]

        delta = (max_return - min_return) / (n - 1)
        if delta > 0:
            returns = np.arange(min_return, max_return + delta, delta)
            vols = list()
            rets = list()
            for r in returns:
                w = self.get_optimal_weights('Vol', constraint=r, constraint_type='Exact')
                if w is not False:
                    result = self.test_weights(w)[:2]
                    rets.append(result[0])
                    vols.append(result[1])
        else:
            rets = [max_return, ]
            vols = [min_vol, ]

        return np.array(vols), np.array(rets)

    def get_optimal_weights(self, target, constraint=None,
                            constraint_type='Exact'):
        if target == 'Sharpe':
            def optimize_function(weights):
                return -self.test_weights(weights)[2]

            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        elif target == 'Vol':
            def optimize_function(weights):
                return self.test_weights(weights)[1]

            cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, ]
            if constraint is not None:
                d = dict()
                if constraint_type == 'Exact':
                    d['type'] = 'eq'
                    d['fun'] = lambda x: self.test_weights(x)[0] - constraint
                    cons.append(d)
                elif constraint_type == 'Bound':
                    d['type'] = 'ineq'
                    d['fun'] = lambda x: self.test_weights(x)[0] - constraint
                    cons.append(d)
                else:
                    msg = 'Value for constraint_type must be either '
                    msg += 'Exact or Bound, not %s' % constraint_type
                    raise ValueError(msg)

        elif target == 'Return':
            def optimize_function(weights):
                return -self.test_weights(weights)[0]

            cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, ]
            if constraint is not None:
                d = dict()
                if constraint_type == 'Exact':
                    d['type'] = 'eq'
                    d['fun'] = lambda x: self.test_weights(x)[1] - constraint
                    cons.append(d)
                elif constraint_type == 'Bound':
                    d['type'] = 'ineq'
                    d['fun'] = lambda x: constraint - self.test_weights(x)[1]
                    cons.append(d)
                else:
                    msg = 'Value for constraint_type must be either '
                    msg += 'Exact or Bound, not %s' % constraint_type
                    raise ValueError(msg)

        else:
            raise ValueError('Unknown target %s' % target)

        bounds = tuple((0, 1) for x in range(self.number_of_assets))
        start = self.number_of_assets * [1. / self.number_of_assets, ]
        result = sco.minimize(optimize_function, start,
                              method='SLSQP', bounds=bounds, constraints=cons)

        if bool(result['success']) is True:
            new_weights = result['x'].round(6)
            return new_weights
        else:
            return False

    def search_sign_changing(self, l, r, f, d):
        if d < 0.000001:
            return (0, 0)
        for x in np.arange(l, r + d, d):
            if f(l) * f(x) < 0:
                ret = (x - d, x)
                return ret

        ret = self.search_sign_changing(l, r, f, d / 2.)
        return ret

    def test_order(self, symbol):
        test = list(self.symbols) + [symbol]
        #name, symbols, shares, cash, start, data_handler
        test_portfolio = MarkowitzMeanVariance('test', test, [1 / len(test) for i in range(len(test))],
                0, self.start_date, self.data_handler)
        test_weights = test_portfolio.get_optimal_weights('Sharpe')
        return (test_portfolio.test_weights(test_weights),
                zip(test_portfolio.symbols, test_weights))

    def size_order(self, weights, prices, investment):
        return [int(investment * weight / price)
                for weight, price in zip(weights, prices)]

    def plot(self):
        # TODO: visualize efficient frontier
        plot_rets = []
        plot_vols = []

        for w in range(500):
            plot_weights = np.random.random(self.number_of_assets)
            plot_weights /= sum(plot_weights)
            r, v, sr = self.test_weights(plot_weights)
            plot_rets.append(r)
            plot_vols.append(v)

        rs = np.array(plot_rets)
        vs = np.array(plot_vols)
        plt.figure(figsize=(15, 9))
        plt.scatter(vs, rs, c=rs / vs, marker='o')
        plt.colorbar(label='Sharpe Ratio')
        plt.plot(*self.get_efficient_frontier(50))
        optimal_weights = self.get_optimal_weights('Sharpe')
        test_returns = self.test_weights(optimal_weights)
        plt.scatter(test_returns[1],
                    test_returns[0],
                    label='Market Portfolio',
#                    marker='x',
                    s=200,
                    c='b',
                    )
        plt.scatter(math.sqrt(self.variance),
                    self.portfolio_return,
                    label='Current Portfolio',
                    s=200,
                    c='r',
                    )
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    from data_handler import WebToDatabase as webdb
    data_handle = webdb()
    stocks = {
         'LGL':  12,
         'RVSB': 22,
         'LLNW': 1,
         }
    portfolio = MarkowitzMeanVariance('Test',
                                      ['LLNW', 'CHS', 'RVSB'],
                                      [10, 10, 10],
                                      100,
                                      start='2016-01-01',
                                      data_handler=data_handle
                                      )
    portfolio.summarize()
