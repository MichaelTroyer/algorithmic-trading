# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 20:59:48 2017

@author: michael

stockScreen:
"""


from bs4 import BeautifulSoup as bs4
import requests


class StockScreen():
    '''Provides methods for rotating sectors and getting lists of stocks.'''
    def __init__(self):

        self.screen = (r'https://secure.marketwatch.com/tools/'
                       r'stockresearch/screener/results.asp')

        self.params = {
            'TradesShareEnable': True,
            'TradesShareMin':    None,
            'TradesShareMax':    None,

            'TradeVolEnable':    True,
            'TradeVolMin':       None,
            'TradeVolMax':       None,

            'BlockEnable':       False,
            'BlockAmt':          None,
            'BlockTime':         None,

            'PERatioEnable':     True,
            'PERatioMin':        None,
            'PERatioMax':        None,

            'MktCapEnable':      True,
            'MktCapMin':         None,
            'MktCapMax':         None,

            'Exchange':          'All',

            'IndustryEnable':    True,
            'Industry':          None,

            'Symbol':            True,
            'CompanyName':       True,
            'Price':             True,
            'Change':            True,
            'ChangePct':         True,
            'Volume':            True,
            'LastTradeTime':     False,
            'FiftyTwoWeekHigh':  False,
            'FiftyTwoWeekLow':   False,
            'PERatio':           True,
            'MarketCap':         True,
            'MoreInfo':          False,

            'SortyBy':           'Volume',
            'SortDirection':     'Descending',
            'ResultsPerPage':    'TwentyFive'
            }

        self.sectors = [
            'Accounting', 'Agriculture', 'Air Freight', 'Air Transport',
            'Alternative Fuel', 'Aluminum', 'Broadcasting', 'Mortgage REITs',
            'Business Services', 'Chemicals', 'Clothing', 'Clothing Retail',
            'Clothing/Textiles', 'Coal', 'Commercial Vehicles', 'Construction',
            'Containers/Packaging', 'Diversified REITs', 'Drug Retail', 'Gold',
            'Electric Utilities', 'Energy', 'Farming', 'Fishing', 'Paper/Pulp',
            'Food Products', 'Food Retail', 'Footwear', 'Fossil Fuels',
            'Funds', 'Furniture', 'Gas Utilities', 'General Mining', 'Tires',
            'General Services', 'Healthcare Provision', 'Healthcare REITs',
            'Housewares', 'Insurance', 'Insurance Brokering', 'Leisure Goods',
            'Investing/Securities', 'Investment Advisors', 'Iron/Steel',
            'Life Insurance', 'Luxury Goods', 'Machinery', 'Finance Companies',
            'Media/Entertainment', 'Mixed Retailing', 'Mobile Machinery',
            'Mortgages', 'Motor Vehicles', 'Multiutilities', 'Oil Extraction',
            'Passenger Airlines', 'Pipeline Transportation', 'Precious Metals',
            'Precision Products', 'Printing', 'Printing/Publishing', 'Hotels',
            'Publishing', 'Railroads', 'Real Estate', 'Reinsurance', 'Retail',
            'Residential REITs', 'Retail REITs', 'Savings Institutions',
            'Securities', 'Shell companies', 'Specialty REITs', 'Utilities',
            'Telecommunication Services', 'Telephone Systems', 'Trucking',
            'Tobacco', 'Tourism', 'Transportation Services', 'Wholesalers',
            'Transportation/Logistics', 'Water Utilities', 'Specialty Retail',
            'Aerospace/Defense', 'Automobiles', 'Automotive', 'Banking',
            'Banking/Credit', 'Biotechnology', 'Beverages/Drinks', 'Software',
            'Computer Services', 'Consumer Finance', 'Consumer Goods',
            'Consumer Services', 'Financial Services', 'Hotels/Restaurants',
            'Industrial Electronics', 'Industrial Goods', 'Internet/Online',
            'Industrial Machinery', 'Networking', 'Pharmaceuticals',
            'Recreational Services', 'Restaurants', 'Retail/Wholesale',
            'Semiconductors', 'Industrial Products', 'Sports Goods',
            'Technical Services', 'Technology']

    def get_top_sectors(self, top_n_sectors=10, low_n_sectors=0):
        '''
        Gets the top n and bottom n sectors measured by the sector mean of the
        product of the volume and the total price change for each of the top
        100 stocks in a given sector, ranked by descending volume.
        '''
        top_sectors = {sector: self.stock_screen(
                Industry=sector,
                ResultsPerPage='OneHundred',
                SortyBy='Volume',
                SortDirection='Descending'
                )
                for sector in self.sectors}

        for sector in top_sectors.keys():
            for s in top_sectors[sector]:
                # replace millions shorthand
                if 'M' in s[5]:
                    s[5] = float(s[5].strip('M')) * 1000000
                # replace billions shorthand
                elif 'B' in s[5]:
                    s[5] = float(s[5].strip('B')) * 1000000000
                else:
                    s[5] = int(s[5].replace(',', ''))
                s[3] = float(s[3].replace(',', ''))

        sector_idx = {sector:
                      (sum([s[5] * s[3] for s in top_sectors[sector]]) /
                       len(top_sectors[sector]))
                      if len(top_sectors[sector]) > 10 else 0
                      for sector in top_sectors.keys()}

        active_sectors = sorted(sector_idx.items(), key=lambda x: -x[1])
        top_sectors = active_sectors[:top_n_sectors]
        low_sectors = active_sectors[-low_n_sectors:] if low_n_sectors else []
        return [s[0] for s in top_sectors] + [s[0] for s in low_sectors]

    def get_stocks(self, urlBase, params):
        response = requests.get(urlBase, params=params)
        soup = bs4(response.text, 'html.parser')
        return self.parse_soup(soup)

    def parse_soup(self, soup):
        stocks = []
        for row in soup.findAll("tr"):
            stocks.append([td.text for td in row.find_all("td")])
        return [stock for stock in stocks if stock]

    def stock_screen(self, **kwargs):
        for keyword, argument in kwargs.items():
            self.params[keyword] = argument
        return self.get_stocks(self.screen, self.params)


if __name__ == '__main__':
    screen = StockScreen()
    top_sectors = screen.get_top_sectors()
    stocks = screen.stock_screen(sector=top_sectors[0])
    print top_sectors
