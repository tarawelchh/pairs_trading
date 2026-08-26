import pandas as pd
import bocd
import numpy as np 
from statsmodels.tsa.ar_model import AutoReg
import yfinance as yf 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import itertools
from statsmodels.tsa.stattools import coint

class PairsTrader:
    def __init__(self, tickerA, tickerB, start="2018-01-01", end="2023-01-01", data=None):
        self.tickerA = tickerA
        self.tickerB= tickerB

        if data is None:
            self.data = yf.download([self.tickerA, self.tickerB], start=start, end=end)['Close'] 
            self.data = self.data.dropna() 

        else:
            self.data = data[[self.tickerA, self.tickerB]].copy()
 
    def calculate_signals(self, window=20, z_score_boundary=2.0):
        #use 30 day look behind to claculate hedge ratio A against B: cov(a,b)/var(b)
        self.data['hedge_ratio'] = self.data[self.tickerA].rolling(window=window).cov(self.data[self.tickerB])/self.data[self.tickerB].rolling(window=window).var()

        #spread = price of A - (hedge ratio*price of B)
        self.data['spread'] = self.data[self.tickerA] - self.data['hedge_ratio']*self.data[self.tickerB]

        #want to compare spread to recent average
        spread_mean = self.data['spread'].rolling(window=window).mean()
        spread_std = self.data['spread'].rolling(window=window).std()
        self.data['z_score'] = (self.data['spread']-spread_mean)/spread_std

        self.data['fast_mean'] = self.data['spread'].rolling(window=10).mean()
        self.data['slow_mean'] = self.data['spread'].rolling(window=60).mean()
        self.data['slow_std'] = self.data['spread'].rolling(window=60).std()

    def cp_basic(self, z_score_boundary=2.0):

        #hard cp
        self.data['changepoint'] = np.abs(self.data['fast_mean']-self.data['slow_mean'])>3*self.data['slow_std']

        #uncertainty
        distance = np.abs(self.data['fast_mean'] - self.data['slow_mean'])
        max_distance = 3 * self.data['slow_std']
        self.data['confidence'] = (1 - (distance / max_distance)).clip(lower=0)

        conditions = [ (self.data['changepoint']==True), (self.data['z_score'] < -z_score_boundary), (self.data['z_score'] > z_score_boundary), (abs(self.data['z_score'])<0.2)] #buy if <-2, sell if >2, exit if close to 0
        choices = [0,1, -1, 0] #1 = buy, -1 = sell, 0 = exit
        self.data['signal'] = np.select(conditions, choices, default=np.nan) #nan tells us to not act

        #dealing with closing data - end of trading day monday becomes tuesday position, ffill overwrites nan with most recent number
        self.data['position'] = self.data['signal'].ffill().shift(1) 

        #bought the spread = buy one, sell the other
        #so profit is +1*bought -1*sold but scaled by hedge
        #if we sold spread -1 times all that, if we flat then 0 times it

        self.data['strategy_return'] = self.data['position']*(self.data[self.tickerA].pct_change()-(self.data['hedge_ratio']*self.data[self.tickerB].pct_change()))
        self.data['strategy_return']*= self.data['confidence'].shift(1)
        #this is as a percentage
        # add 1 to the daily returns and calculate t,he cumulative product
        self.data['cumulative_return'] = (1 + self.data['strategy_return']).cumprod()

    def cp_bayes(self, window=20, z_score_boundary=2.0):
        model = bocd.BayesianOnlineChangePointDetection(
            bocd.ConstantHazard(250), # Prior: Expecting a regime shift roughly once a trading year
            bocd.StudentT(mu=0, kappa=1, alpha=1, beta=1) 
        )
        
        cp_probabilities = []
        
        # Process the spread sequentially (mimicking live tick data)
        for current_spread in self.data['spread'].dropna():
            model.update(current_spread) 
            
            # Extract the probability that a change point literally just happened (run length = 0)
            cp_prob = model.rt[0] 
            cp_probabilities.append(cp_prob)
            
        # (Pad cp_probabilities with NaNs at the beginning to match your original dataframe length)
        padding_length = len(self.data) - len(cp_probabilities)
        padded_cp_probabilities = [np.nan] * padding_length + cp_probabilities
        self.data['cp_probability'] = padded_cp_probabilities
        
        # Redefine your confidence scalar mathematically
        # E.g., if there is a 95% chance a regime shift just occurred, confidence drops to 5%
        self.data['confidence'] = 1.0 - self.data['cp_probability']

        conditions = [ (self.data['cp_probability']>0.95), (self.data['z_score'] < -z_score_boundary), (self.data['z_score'] > z_score_boundary), (abs(self.data['z_score'])<0.2)] 
        choices = [0, 1, -1, 0] # 0 = exit on regime shift, 1 = buy, -1 = sell, 0 = exit
        self.data['signal'] = np.select(conditions, choices, default=np.nan) 
        
        self.data['position'] = self.data['signal'].ffill().shift(1) 
        self.data['strategy_return'] = self.data['position']*(self.data[self.tickerA].pct_change()-(self.data['hedge_ratio']*self.data[self.tickerB].pct_change()))
        self.data['strategy_return']*= self.data['confidence'].shift(1)
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

# test = PairsTrader("2018-01-01", "2023-01-01", "AAPL", "MSFT")
# test.fetch_data()
# test.calculate_signals()
# test.plot_performance()
# test.print_metrics()

start = "2018-01-01"
end = "2023-01-01"
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "V", "MA", "JPM", "BAC"]
data = yf.download(tickers, start=start, end=end)['Close'] 
returns = data.pct_change().dropna()

train_returns = returns.loc[:'2020-12-31']
test_returns = returns.loc['2021-01-01':]

scaler = StandardScaler()
scaled_train = scaler.fit_transform(train_returns)
pca = PCA(n_components=1)
pca1_train = pca.fit_transform(scaled_train)

variance = pca.explained_variance_ratio_[0] * 100
print(f"\nTrain PC1 explains {variance:.2f}% of variability")

sp500 = yf.download("^GSPC", start=start, end="2020-12-31")['Close']
sp500_returns = sp500.pct_change().dropna()
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(train_returns.index, pca1_train.cumsum(), color='red', label='Train PC1')
ax1.set_ylabel('Train PC1 Cumulative', color='red')
ax2 = ax1.twinx()
ax2.plot(sp500_returns.index, sp500_returns.cumsum(), color='blue', label='S&P 500')

ax2.set_ylabel('S&P 500 Cumulative', color='blue')
plt.title("In-Sample PC1 vs S&P 500")
plt.show()

loaded_returns_train = pca.inverse_transform(pca1_train)
residuals_train = scaled_train - loaded_returns_train
unscaled_residuals_train = residuals_train * scaler.scale_
residuals_df_train = pd.DataFrame(unscaled_residuals_train, index=train_returns.index, columns=train_returns.columns)
train_prices = (1 + residuals_df_train).cumprod()

scaled_test = scaler.transform(test_returns)
pca1_test = pca.transform(scaled_test)
loaded_returns_test = pca.inverse_transform(pca1_test)
residuals_test = scaled_test - loaded_returns_test
unscaled_residuals_test = residuals_test * scaler.scale_
residuals_df_test = pd.DataFrame(unscaled_residuals_test, index=test_returns.index, columns=test_returns.columns)
test_prices = (1 + residuals_df_test).cumprod()

cointegration_results = []
pairs = itertools.combinations(train_prices.columns, 2)

for ticker1, ticker2 in pairs:
    series1 = train_prices[ticker1]
    series2 = train_prices[ticker2]
    _, p_value, _ = coint(series1, series2)
    cointegration_results.append((ticker1, ticker2, p_value))

coint_df = pd.DataFrame(cointegration_results, columns=['Stock 1', 'Stock 2', 'P-Value'])
coint_df = coint_df.sort_values(by='P-Value')
print("\n--- Top 3 Most Cointegrated Pairs (In-Sample) ---")
print(coint_df.head(3))

windows = [10, 15, 20, 25, 30, 40, 50, 60]
z_entries = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
combinations = list(itertools.product(windows, z_entries))
grid_results = []

for w, z in combinations:
    trader = PairsTrader("V", "MA", data=train_prices)
    trader.calculate_signals(window=w, z_score_boundary=z)
    trader.cp_basic(z_score_boundary=z) 
    
    daily_returns = trader.data['strategy_return'].dropna()
    
    if daily_returns.std() == 0:
        sharpe = 0
        total_ret = 0
    else:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        total_ret = trader.data['cumulative_return'].iloc[-1] - 1
        
    grid_results.append({'Window': w, 'Z-Entry': z, 'Sharpe': sharpe, 'Return': total_ret})

results_df = pd.DataFrame(grid_results).sort_values(by='Sharpe', ascending=False)
print("\n--- Top 3 Combinations (In-Sample Training) ---")
print(results_df.head(3))

heatmap_data = results_df.pivot(index='Window', columns='Z-Entry', values='Sharpe')
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', center=0, fmt='.2f')
plt.title("In-Sample Parameter Stability (Sharpe Ratio)")
plt.xlabel("Z-Score Entry Threshold")
plt.ylabel("Moving Average Window")
plt.show()

# 7. Out-of-Sample Blind Test (A/B Comparison)
print("\n--- Out-of-Sample Blind Test (2021-2022) ---")

# --- METHOD A: Basic CPD ---
trader_basic = PairsTrader("GOOGL", "JPM", data=test_prices)
trader_basic.calculate_signals(window=40, z_score_boundary=2.0) 
trader_basic.cp_basic(z_score_boundary=2.0)

basic_returns = trader_basic.data['strategy_return'].dropna()
basic_sharpe = (basic_returns.mean() / basic_returns.std()) * np.sqrt(252)
basic_total_ret = trader_basic.data['cumulative_return'].iloc[-1] - 1

print("\n[Basic Moving Average CPD]")
print(f"Total Return: {basic_total_ret:.2%}")
print(f"Sharpe Ratio: {basic_sharpe:.2f}")

# --- METHOD B: Bayesian CPD ---
trader_bayes = PairsTrader("GOOGL", "JPM", data=test_prices)
trader_bayes.calculate_signals(window=40, z_score_boundary=2.0) 
trader_bayes.cp_bayes(window=40, z_score_boundary=2.0) # Triggering the Bayesian method

bayes_returns = trader_bayes.data['strategy_return'].dropna()
bayes_sharpe = (bayes_returns.mean() / bayes_returns.std()) * np.sqrt(252)
bayes_total_ret = trader_bayes.data['cumulative_return'].iloc[-1] - 1

print("\n[Bayesian Online CPD]")
print(f"Total Return: {bayes_total_ret:.2%}")
print(f"Sharpe Ratio: {bayes_sharpe:.2f}")