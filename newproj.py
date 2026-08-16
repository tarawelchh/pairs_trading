import yfinance as yf
import pandas as pd
import itertools
from statsmodels.tsa.stattools import coint
from statsmodels.regression.rolling import RollingOLS
from statsmodels.regression import linear_model
import statsmodels as sm
import matplotlib.pyplot as plt
import numpy as np
import bocd

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "V", "MA", "JPM", "BAC"]
start = "2018-01-01"
end = "2023-01-01"
split_date = "2019-12-31"
data = yf.download(tickers, start=start, end=end)['Close'] 
data = data.dropna()


training_data = data.loc[:split_date]
trading_data = data.loc[split_date:].copy()
combos = (itertools.combinations(tickers,2))
pvalues = []

# Before we can calculate a spread, we need to find two stocks that are actually tethered together. 
# We do this using the Engle-Granger cointegration test.

for a, b in combos:
    series_a = training_data[a]
    series_b = training_data[b]
    _, pval, _ = coint(series_a, series_b)
    pvalues.append((a,b,pval))

pvaldf = pd.DataFrame(pvalues)
pvaldf.columns = ["Stock A", "Stock B", "p-value"]
pvaldf = pvaldf.sort_values("p-value", ascending=True)
print(pvaldf)

# Now that we know which two stocks move together, we need to create the "elastic band" (the spread).
# In your original code, you used Cov(A,B) / Var(B). This is flawed because it forces the spread to be centered 
# at zero, completely ignoring the absolute price difference between the stocks (the intercept, alpha).
# Instead, the professional standard is to run a Rolling Ordinary Least Squares (OLS) regression.
# The mathematical relationship is: {Price}_A = \alpha + \beta \times \{Price}_B
# Where: \beta (Slope / Hedge Ratio): How many shares of B we short for every 1 share of A.
# \alpha (Intercept): The baseline price gap between them.
# Once we know $\alpha$ and $\beta$, our stationary spread is simply the residual (the error of the regression):


chosen_a, chosen_b = pvaldf.iloc[0, 0:2]
#price_a = alpha + beta*price_b

long_window_size = 60
ols = RollingOLS(trading_data[chosen_a], sm.tools.add_constant(trading_data[chosen_b]), long_window_size)
model = ols.fit() 
alpha = model.params['const'] #NaNs for first 59 days as window is 60 days
beta = model.params[chosen_b]

# Spread = {Price}_A - (\beta \times \{Price}_B + \alpha)
spread = trading_data[chosen_a] - (beta * trading_data[chosen_b] + alpha )
plt.figure(figsize=(10,5))
plt.axhline(0, linestyle="--")
spread.plot(color="blue")
plt.show()


# We need to mathematically calculate exactly how many days it takes for our elastic band to snap halfway 
# back to zero. This is called the Half-Life, derived from the Ornstein-Uhlenbeck (OU) process.
# Here is the secret: to find the half-life, we just run another simple linear regression, but this time on the spread itself.
# If a spread is mean-reverting, then the change in the spread today should depend on where 
# the spread was yesterday. If yesterday's spread was huge, today's change should be a big negative move (snapping back).

#calculate the daily change in spread 
#regress against the previous days spread
spread_clean = spread.dropna()
print(spread_clean.head())
change = np.array(spread_clean.diff(1).dropna())
x = np.array(spread_clean.shift().dropna())

ols_spread = linear_model.OLS(change, sm.tools.add_constant(x))
results = ols_spread.fit()
print(results.summary())

#negative x1 shows mean reverting
#half-life = -ln2/gradient 

half_life = -np.log(2)/results.params[1]
print(half_life)
#half life 11 days

short_window_size=(np.floor(half_life)).astype(int)

spread_mean = spread.rolling(window=short_window_size).mean()
spread_std = spread.rolling(window=short_window_size).std() + 1e-8 #prevent division by zero
z_score = (spread - spread_mean)/spread_std
trading_data["z_score"] = z_score
print(z_score.tail())

# We expect a structural break roughly once every 250 trading days (1 year)
# We model the spread using a Student-T distribution (handles fat tails better than Gaussian)
bayes_model = bocd.BayesianOnlineChangePointDetection(
    bocd.ConstantHazard(250), #prior is one changepoint per year
    bocd.StudentT(mu=0, kappa=1, alpha=1, beta=1) #financial market has fat tails
)

coint_p_vals = []
padding = [np.nan] * 60 

for i in range(60, len(trading_data)):
    # Slice a rolling window that moves forward with 'i'
    view = trading_data[[chosen_a, chosen_b]].iloc[i-60:i] 
    
    # Pass the two distinct columns into the cointegration test
    _, pvalview, _ = coint(view[chosen_a], view[chosen_b])
    coint_p_vals.append(pvalview)

# Combine the padding and the values, then add to the dataframe
trading_data["coint_p_vals"] = padding + coint_p_vals

z_score_clean = z_score.dropna()
cp_probs = []
for value in z_score_clean:
    bayes_model.update(value)
    cp_probs.append(bayes_model.rt[0])

padding = np.repeat(np.nan, (len(trading_data) - len(cp_probs)))
padded_probs = np.append(padding,cp_probs)
trading_data["cp_probs"] = padded_probs
print(trading_data.tail())

# 1. The Lock (Absolute Priority)
brake_triggers = trading_data["cp_probs"].diff() < 0
trading_data.loc[brake_triggers, "locked_down"] = True

# 2. The Key (Only allowed to unlock if the brake is NOT triggering today)
# We wrap both conditions in parentheses and use '&' for 'AND'
valid_coint = (trading_data["coint_p_vals"] < 0.05) & (~brake_triggers)
trading_data.loc[valid_coint, "locked_down"] = False

# 3. The Flip-Flop
trading_data["locked_down"] = trading_data["locked_down"].ffill().fillna(False)

conditions = [(trading_data["locked_down"]==True), (trading_data["cp_probs"].diff() <0), (trading_data["z_score"]< -2.0), (trading_data["z_score"]>2.0), (abs(trading_data["z_score"])<0.2)]
choices = [0, 0, 1, -1, 0]
trading_data["position"] = np.select(conditions, choices, default=np.nan)
trading_data["position"] = trading_data["position"].ffill().shift(1)

trading_data["price_ratio"] = trading_data[chosen_a]/data[chosen_b]
trading_data["ratio_return"] = trading_data["price_ratio"].pct_change()
trading_data["strategy_return"] = trading_data["position"] * trading_data["ratio_return"]
trading_data["cumulative_return"] = (1 + trading_data["strategy_return"]).cumprod()

total_ret = (trading_data["cumulative_return"].iloc[-1] - 1) * 100
print(f"Final Strategy Return with BOCD Emergency Brake: {total_ret:.2f}%")

changepoints = trading_data[trading_data["cp_probs"]==1.0]
print(changepoints)
trading_data.to_csv('my_pairs_trading_results.csv')