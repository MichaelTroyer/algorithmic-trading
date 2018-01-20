# -*- coding: utf-8 -*-

"""
Created on Sun Oct 22 22:04:30 2017

@author: michael
"""


from __future__ import division
from abc import ABCMeta, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import datetime
import traceback


class Strategy(object):
    """
    Strategy is an abstract base class providing an interface for
    all subsequent (inherited) strategy objects.

    Enforces: get_DataFrame() method for retrieving stock data as a
    Pandas DataFrame object.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def process_symbol(self):
        """
        Processes the stock ticker and returns the signal with
        optional summary and plot
        """
        raise NotImplementedError("Should implement plot()")


class MovingAverageConvergenceDivergence(Strategy):
    def __init__(self, periods, start_date, data_handler):

        self.periods = periods
        self.start_date = start_date
        self.data_handler = data_handler
        self.benchmark_df = data_handler.get_DataFrame('SPY', start_date)
        self.benchmark_df['MarketLogRet'] = np.log(self.benchmark_df['Close'] / self.benchmark_df['Close'].shift(1))
        self.benchmark_df['CumuMarketRet'] = self.benchmark_df['MarketLogRet'].cumsum().apply(np.exp)
        self.benchmark = self.benchmark_df.CumuMarketRet.values[-1]

        self.short_ema = 'Close_{}-EMA'.format(self.periods[0])
        self.long_ema = 'Close_{}-EMA'.format(self.periods[1])
        self.macd_ema = 'MACD_{}_EMA'.format(self.periods[2])
        self.volume_ema = 'Volume_{}_EMA'.format(self.periods[2])

    def process_symbol(self, symbol, summarize=True, plot=True):
        try:
            self.symbol = symbol
            self.data = self.data_handler.get_DataFrame(symbol, self.start_date)
            self.data[self.short_ema] = self.data.Close.ewm(span=self.periods[0]).mean()
            self.data[self.long_ema] = self.data.Close.ewm(span=self.periods[1]).mean()
            self.data['MACD'] = self.data[self.short_ema] - self.data[self.long_ema]
            self.data[self.macd_ema] = self.data.MACD.ewm(span=self.periods[2]).mean()
            self.data['MACD_Delta'] = self.data.MACD - self.data[self.macd_ema]

            self.data[self.volume_ema] = self.data.Volume.ewm(span=self.periods[2]).mean()
            self.data['Volume_EMA_upper'] = self.data[self.volume_ema] * 1.5
            self.data['Volume_EMA_lower'] = self.data[self.volume_ema] * 0.5

            self.crossover_up = []
            self.crossover_dw = []
            prev_pos = -1
            i = 0
            for row in self.data.itertuples():
                row_date = row.__getattribute__('Index')
                row_close = row.__getattribute__('Close')
                row_macd = row.__getattribute__('MACD')
                macd_dlt = row.__getattribute__('MACD_Delta')
                if prev_pos == 1:
                    if row_macd < macd_dlt < 0:
                        pos = -1
                        self.crossover_dw.append((row_date, row_close))
                    else:
                        pos = 1
                else:
                    if row_macd > macd_dlt > 0:
                        pos = 1
                        self.crossover_up.append((row_date, row_close))
                    else:
#                        pos = -1
                        pos = 0
                prev_pos = pos
                self.data.loc[row_date, 'Position'] = pos
                i += 1

            self.data['MarketLogRet'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
            self.data['StrategyLogRet'] = self.data['Position'].shift(1) * self.data['MarketLogRet']

            self.data['CumuMarketRet'] = self.data['MarketLogRet'].cumsum().apply(np.exp)
            self.data['CumuStrategyRet'] = self.data['StrategyLogRet'].cumsum().apply(np.exp)

            self.AnnualizedMarketLogRet = self.data['MarketLogRet'].mean() * 252
            self.AnnualizedStrategyLogRet = self.data['StrategyLogRet'].mean() * 252

            self.AnnualizedMarketVolatility = self.data['MarketLogRet'].std() * 252 ** 0.5
            self.AnnualizedStrategyVolatility = self.data['StrategyLogRet'].std() * 252 ** 0.5

            self.return_ratio = self.AnnualizedStrategyLogRet / self.AnnualizedMarketLogRet

            self.data['K'] = (100 *
                     ((self.data.Close - self.data.Low.rolling(window=14,center=False).min()) /
                      (self.data.High.rolling(window=14,center=False).max() -
                       self.data.Low.rolling(window=14,center=False).min()))
                     )

            self.data['D'] = self.data.K.rolling(window=3,center=False).mean()
            slow = True
            if slow:
                # If slow stochaastic, roll ma_period window again
                self.data['D'] = self.data.D.rolling(window=3,center=False).mean()

            if summarize:
                self._summary()
            if plot:
                self._plot()
            return self._signal()

        except Exception as e:
            raise Exception, '[+] Strategy Error - [{}]\n{}'.format(e, traceback.format_exc())

    def _signal(self):
        last_macd = self.data.MACD.tolist()[-1]
        last_macd_ema = self.data[self.macd_ema].tolist()[-1]

        if (self.AnnualizedStrategyLogRet > self.AnnualizedMarketLogRet and
            self.AnnualizedStrategyLogRet > self.benchmark):
            if last_macd > last_macd_ema > 0:
                return 'BUY'
            elif last_macd < last_macd_ema < 0:
                return 'SELL'
            else:
                return 'NONE'
        else:
            return 'FAIL'

    def _period_metrics(self, buy_date_price_tuples, sell_date_price_tuples):

        wins = []
        loss = []

        # If last transaction was buy, set today as sell date for returns calc
        if len(buy_date_price_tuples) > len(sell_date_price_tuples):
            sell_date_price_tuples.append(
                (datetime.datetime.today(), self.data.Close[-1]))

        periods = zip(buy_date_price_tuples, sell_date_price_tuples)

        for (buy_date, buy_price), (sell_date, sell_price) in periods:
            period_length = (sell_date - buy_date).days
            period_return = (sell_price - buy_price)
            if period_return > 0:
                wins.append((period_length, period_return))
            else:
                loss.append((period_length, period_return))

        results = [wins, loss]
        n_wins = len(wins)
        n_loss = len(loss)
        sum_wins = sum([w[1] for w in wins])
        sum_loss = sum([w[1] for w in loss])
        average_wins = sum_wins / n_wins if n_wins else 0.0
        average_loss = sum_loss / n_loss if n_loss else 0.0
        net_returns = sum_wins + sum_loss
        market_net = self.data.Close[-1] - self.data.Close[0]

        print 'N Winning Periods           : {}'.format(n_wins)
        print 'N Losing Periods            : {}'.format(n_loss)
        print 'Average Win                 : {:.4}'.format(average_wins)
        print 'Average Loss                : {:.4}'.format(average_loss)
        print 'Market Net Returns          : {:.4}'.format(market_net)
        print 'Strategy Net Returns        : {:.4}'.format(net_returns)
        print
        print 'Wins:'
        for days, price in wins:
            print 'Days: {:>4}     Earnings:    : {:.4}'.format(days, price)
        print
        print 'Losses:'
        for days, price in loss:
            print 'Days: {:>4}     Earnings:    : {:.4}'.format(days, price)
#
#        return {
#                'Results': results,
#                'N Wins': n_wins,
#                'N Loss': n_loss,
#                'Average Win': average_wins,
#                'Average Loss': average_loss,
#                'Net Returns': net_returns
#                }

    def _summary(self):
        print
        print self.symbol.center(70, '-')
        print
        print 'Moving Average Convergence Divergence [{}, {}, {}]'.format(*self.periods).center(70)
        print
        print 'Signal                      : {}'.format(self._signal())
        print 'Position                    : {}'.format(self.data.Position.values[-1])
        print 'Current Price               : {}'.format(self.data.Close.values[-1])
        print 'Current Price Short EMA     : {}'.format(self.data[self.short_ema].values[-1])
        print 'Current Price Long EMA      : {}'.format(self.data[self.long_ema].values[-1])
        print 'Current Volume              : {}'.format(self.data.Volume.values[-1])
        print 'Current Volume Long EMA     : {}'.format(int(self.data[self.volume_ema].values[-1]))
        print
        print 'Annual Market Log Returns   : {:.4}'.format(self.AnnualizedMarketLogRet)
        print 'Annual Market Volatility    : {:.4}'.format(self.AnnualizedMarketVolatility)
        print 'Annual Strategy Log Returns : {:.4}'.format(self.AnnualizedStrategyLogRet)
        print 'Annual Strategy Volatility  : {:.4}'.format(self.AnnualizedStrategyVolatility)
        print 'Cumulative Market Returns   : {:.4}'.format(self.data.CumuMarketRet.values[-1])
        print 'Cumulative Strategy Returns : {:.4}'.format(self.data.CumuStrategyRet.values[-1])
        print 'Benchmark [SPY] Returns     : {:.4}'.format(self.benchmark)
        print
        self._period_metrics(self.crossover_up, self.crossover_dw)

    def _plot(self):
        fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(20, 15),
                                 gridspec_kw = {'width_ratios':[2, 1]})
#        fig.tight_layout()

        self.data.Close.plot(ax=axes[0][0], alpha=0.3, rot=45)
        self.data[[self.short_ema, self.long_ema]].plot(ax=axes[0][0], rot=45)
        up_xs, up_ys = zip(*self.crossover_up)
        dw_xs, dw_ys = zip(*self.crossover_dw)
        axes[0][0].scatter(up_xs, up_ys, c='b', label='Buy')
        axes[0][0].scatter(dw_xs, dw_ys, c='r', label='Sell')
        axes[0][0].legend()

        self.data[['MACD', self.macd_ema]].plot(ax=axes[1][0])
        self.data.MACD_Delta.plot(kind='area', alpha=0.3, stacked=False, ax=axes[1][0], rot=45)

        self.data.Volume.plot(ax=axes[2][0], alpha=0.3)
        self.data[[self.volume_ema, 'Volume_EMA_upper', 'Volume_EMA_lower']].plot(ax=axes[2][0])

        self.data.D.plot(ax=axes[3][0], label='Oscillator')
        axes[3][0].axhline(y=80, xmin=0, xmax=1, color='r')
        axes[3][0].axhline(y=20, xmin=0, xmax=1, color='r')
        axes[3][0].legend()

        self.data[['CumuMarketRet', 'CumuStrategyRet']].plot(ax=axes[4][0], rot=45)

        # Plot the short-term
        self.short_data = self.data.tail(30)

        self.short_data.Close.plot(ax=axes[0][1], alpha=0.3, rot=45)
        self.short_data[[self.short_ema, self.long_ema]].plot(ax=axes[0][1], rot=45)
        axes[0][1].legend()

        self.short_data[['MACD', self.macd_ema]].plot(ax=axes[1][1])
        self.short_data.MACD_Delta.plot(kind='area', alpha=0.3, stacked=False, ax=axes[1][1], rot=45)

        self.short_data.Volume.plot(ax=axes[2][1], alpha=0.3)
        self.short_data[[self.volume_ema, 'Volume_EMA_upper', 'Volume_EMA_lower']].plot(ax=axes[2][1])

        self.short_data.D.plot(ax=axes[3][1], label='Oscillator')
        axes[3][1].axhline(y=80, xmin=0, xmax=1, color='r')
        axes[3][1].axhline(y=20, xmin=0, xmax=1, color='r')
        axes[3][1].legend()

        self.short_data[['CumuMarketRet', 'CumuStrategyRet']].plot(ax=axes[4][1], rot=45)
        fig.subplots_adjust(hspace=0.1)
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1][:-1]], visible=False)
        for a in fig.axes: a.set_xlabel(' ')

        axes[0][1].legend().set_visible(False)
        axes[1][1].legend().set_visible(False)
        axes[2][1].legend().set_visible(False)
        axes[3][1].legend().set_visible(False)
        axes[4][1].legend().set_visible(False)

        axes[0][0].set_ylabel('Closing Price')
        axes[1][0].set_ylabel('Price Difference')
        axes[2][0].set_ylabel('Trade Volume')
        axes[3][0].set_ylabel('Stochastic Oscillator')
        axes[4][0].set_ylabel('Cumulative Returns')

        plt.show()


if __name__ == '__main__':
    from data_handler import WebToDatabase as qdb
    data_handle = qdb()
    macd = MovingAverageConvergenceDivergence((5, 25, 25), '2017-01-01', data_handle)
#    print macd.process_symbol('AAPL')
    print macd.process_symbol('SPYG')
