import pandas as pd
import numpy as np 
from statsmodels.tsa.ar_model import AutoReg
import yfinance as yf 
import matplotlib.pyplot as plt

class PairsTrader:
    def __init__(self, start, end, tickerA, tickerB):
        self.start = start
        self.end = end 
        self.tickerA = tickerA
        self.tickerB= tickerB

    def fetch_data(self):
        self.data = yf.download([self.tickerA, self.tickerB], start=self.start, end=self.end)['Close']  
        #yfinance gives pandas dataframe
        self.data = self.data.dropna() #remove rows with missing data 
    
    def calculate_signals(self, window=30):
        #use 30 day look behind to claculate hedge ratio A against B: cov(a,b)/var(b)
        self.data['hedge_ratio'] = self.data[self.tickerA].rolling(window=window).cov(self.data[self.tickerB])/self.data[self.tickerB].rolling(window=window).var()

        #spread = price of A - (hedge ratio*price of B)
        self.data['spread'] = self.data[self.tickerA] - self.data['hedge_ratio']*self.data[self.tickerB]

        #want to compare spread to recent average
        spread_mean = self.data['spread'].rolling(window=window).mean()
        spread_std = self.data['spread'].rolling(window=window).std()
        self.data['z_score'] = (self.data['spread']-spread_mean)/spread_std

        conditions = [(self.data['z_score'] < -2), (self.data['z_score'] > 2), (abs(self.data['z_score'])<0.2)] #buy if <-2, sell if >2, exit if close to 0
        choices = [1, -1, 0] #1 = buy, -1 = sell, 0 = exit
        self.data['signal'] = np.select(conditions, choices, default=np.nan) #nan tells us to not act

        #dealing with closing data - end of trading day monday becomes tuesday position, ffill overwrites nan with most recent number
        self.data['position'] = self.data['signal'].ffill().shift(1) 

        #bought the spread = buy one, sell the other
        #so profit is +1*bought -1*sold but scaled by hedge
        #if we sold spread -1 times all that, if we flat then 0 times it

        self.data['strategy_return'] = self.data['position']*(self.data[self.tickerA].pct_change()-(self.data['hedge_ratio']*self.data[self.tickerB].pct_change()))
        #this is as a percentage
        # add 1 to the daily returns and calculate the cumulative product
        self.data['cumulative_return'] = (1 + self.data['strategy_return']).cumprod()
    
    def plot_performance(self):
        self.data['cumulative_return'].plot()
        plt.show()

    def print_metrics(self):
        final_return = (self.data['cumulative_return'].iloc[-1]-1)*100
        sharpe = (self.data['strategy_return'].mean()/self.data['strategy_return'].std())*np.sqrt(252)
        rolling_max = self.data['cumulative_return'].cummax()
        drawdown = (self.data['cumulative_return'] / rolling_max) - 1.0
        max_drawdown = drawdown.min() * 100

        print("--- Strategy Performance ---")
        print(f"Total Return: {final_return:.2f}%")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")

test = PairsTrader("2018-01-01", "2023-01-01", "V", "MA")
test.fetch_data()
test.calculate_signals()
test.plot_performance()
test.print_metrics()