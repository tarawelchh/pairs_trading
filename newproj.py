import yfinance as yf
import pandas as pd
import itertools
from statsmodels.tsa.stattools import coint
from statsmodels.regression.rolling import RollingOLS
from statsmodels.regression import linear_model
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import bocd

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "V", "MA", "JPM", "BAC"]
start = "2018-01-01"
end = "2023-01-01"
split_date = "2019-12-31"
data = yf.download(tickers, start=start, end=end)['Close'] 
data = data.dropna()
transaction_cost = 0.0005

training_data = data.loc[:split_date]
trading_data = data.loc[split_date:].copy()
combos = (itertools.combinations(tickers,2))

def engle_granger(combos, data):
    pvalues = []
    for a, b in combos:
        series_a = data[a]
        series_b = data[b]
        _, pval, _ = coint(series_a, series_b)
        pvalues.append((a,b,pval))

    pvaldf = pd.DataFrame(pvalues)
    pvaldf.columns = ["Stock A", "Stock B", "p-value"]
    pvaldf = pvaldf.sort_values("p-value", ascending=True)
    return pvaldf

def calculate_spread(window, stock_a, stock_b, data):
    ols = RollingOLS(data[stock_a], sm.add_constant(data[stock_b]), window)
    model = ols.fit() 
    alpha = model.params['const']
    beta = model.params[stock_b]
    spread = data[stock_a] - (beta * data[stock_b] + alpha )
    return(spread, beta)

def calculate_halflife(spread):
    change = spread.diff()
    x = spread.shift()
    x_c = sm.add_constant(x)
    ols_spread = sm.OLS(change, x_c, missing='drop')
    results = ols_spread.fit()
    gradient = results.params.iloc[1]
    if gradient >= 0:
        print("Gradient is positive - series is not mean-reverting")
        return np.nan
    half_life = -np.log(2)/gradient #maybe add bounds in case it gets close to 0 as his will approach inf
    return half_life

def z_scores(spread, window):
    spread_mean = spread.rolling(window=window).mean()
    spread_std = spread.rolling(window=window).std() + 1e-8 #prevent division by zero
    z_score = (spread - spread_mean)/spread_std
    return z_score

def bayesian_cp(data, z_score, stock_a, stock_b, long_window_size):
    df = data.copy()
    
    bayes_model = bocd.BayesianOnlineChangePointDetection(
        bocd.ConstantHazard(63), #prior is one cp per quarter
        bocd.StudentT(mu=0, kappa=1, alpha=1, beta=1) 
    )

    coint_p_vals = [np.nan] * long_window_size 
    arr_a = df[stock_a].values
    arr_b = df[stock_b].values

    for i in range(long_window_size, len(df)):
        view_a = arr_a[i-long_window_size : i]
        view_b = arr_b[i-long_window_size : i]
        
        if np.isnan(view_a).any() or np.isnan(view_b).any():
            coint_p_vals.append(np.nan)
        else:
            _, pvalview, _ = coint(view_a, view_b)
            coint_p_vals.append(pvalview)
            
    df["coint_p_vals"] = coint_p_vals
    z_score_clean = z_score.dropna()
    cp_day = []
    
    for value in z_score_clean:
        bayes_model.update(value)
        cp_day.append(bayes_model.rt[0])
        
    df["cp_day"] = pd.Series(cp_day, index=z_score_clean.index)
    return df

def generate_signals_and_returns(data, z_entry, z_exit=0.2, coint_thresh=0.05):
    df = data.copy()
    
    brake_triggers = df["cp_day"].diff() < 0
    df.loc[brake_triggers, "locked_down"] = True
    #unlock if cointegration and brake isn't triggering today
    valid_coint = (df["coint_p_vals"] < coint_thresh) & (~brake_triggers)
    df.loc[valid_coint, "locked_down"] = False

    df["locked_down"] = df["locked_down"].ffill().fillna(False)
    conditions = [
        (df["locked_down"] == True), 
        (df["cp_day"].diff() < 0), 
        (df["z_score"] < -z_entry), 
        (df["z_score"] > z_entry), 
        (abs(df["z_score"]) < z_exit)
    ]
    choices = [0, 0, 1, -1, 0]
    
    df["position"] = np.select(conditions, choices, default=np.nan)
    df["position"] = df["position"].ffill().shift(1)
    return df

def calculate_returns(stock_a, stock_b, beta, data=trading_data):
    data["delta_A"] = data[stock_a].diff()
    data["delta_B"] = data[stock_b].diff()

    data["beta_shifted"] = beta.shift(1)
    data["gross_exposure"] = data[stock_a].shift(1) + (abs(data["beta_shifted"]) * data[stock_b].shift(1))

    # PnL = Delta Price_A - (beta * delta Price_B)
    transaction_cost_dollars = transaction_cost * data["gross_exposure"] * abs(data["position"].diff())
    data["spread_pnl"] = data["delta_A"] - (data["beta_shifted"] * data["delta_B"]) - transaction_cost_dollars
    data["strategy_return"] = data["position"] * (data["spread_pnl"] / data["gross_exposure"])
    data["cumulative_return"] = (1 + data["strategy_return"].fillna(0)).cumprod()
    return data

def parameter_tuning(data, chosen_a, chosen_b):
    ols_windows = range(20,40, 2) 
    z_entries = np.arange(0.5, 2.0, 0.1) 
    results = []

    for win in ols_windows:
        df_base = data.copy()

        spread, beta = calculate_spread(win, chosen_a, chosen_b, df_base)
        half_life = calculate_halflife(spread)
        
        if pd.isna(half_life) or half_life < 2:
            continue
            
        short_w = int(np.floor(half_life))
        df_base["z_score"] = z_scores(spread, short_w)
        df_base = bayesian_cp(df_base, df_base["z_score"], chosen_a, chosen_b, win)
        
        for z in z_entries:
            df = df_base.copy() 
            
            df = generate_signals_and_returns(df, z_entry=z)
            df = calculate_returns(chosen_a, chosen_b, beta, data=df)
            
            total_ret = (df["cumulative_return"].iloc[-1] - 1) * 100
            daily_vol = df['strategy_return'].std()
            sharpe = (df['strategy_return'].mean() / daily_vol) * np.sqrt(252) if daily_vol > 0 else 0
            rolling_max = df["cumulative_return"].cummax()
            drawdown = (df["cumulative_return"] - rolling_max) / rolling_max
            max_dd = abs(drawdown.min())
            calmar = (total_ret / 100) / (max_dd + 1e-8)
            results.append({
                "Window": win, 
                "Z-Entry": round(z, 2), 
                "Return (%)": total_ret, 
                "Sharpe": sharpe,
                "Calmar": calmar
            })

    results_df = pd.DataFrame(results)
    return results_df

def plot_optimization_surface(df, metric="Calmar"):
    pivot_table = df.pivot(index="Window", columns="Z-Entry", values=metric)
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        pivot_table, 
        annot=True,          
        cmap="coolwarm",     
        center=0,            
        fmt=".2f",           
        cbar_kws={'label': metric}
    )
    
    plt.title(f"Parameter Surface based on {metric}")
    plt.xlabel("Z-Score")
    plt.ylabel("Window")

    plt.show()

def print_strategy_summary(df):
    total_return = (df["cumulative_return"].iloc[-1] - 1) * 100
    
    daily_vol = df['strategy_return'].std()
    sharpe = (df['strategy_return'].mean() / daily_vol) * np.sqrt(252) if daily_vol > 0 else 0

    rolling_max = df["cumulative_return"].cummax()
    drawdown = (df["cumulative_return"] - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    
    total_trades = (df["position"].diff().fillna(0) != 0).sum()
    days_locked_down = (df["locked_down"] == True).sum()
    total_changepoints = (df["cp_day"].diff() < 0).sum()
    
    print("\n" + "="*40)
    print("      STRATEGY PERFORMANCE SUMMARY")
    print("="*40)
    print(f"Total Cumulative Return: {total_return:>8.2f}%")
    print(f"Sharpe Ratio:            {sharpe:>8.2f}")
    print(f"Maximum Drawdown:        {max_drawdown:>8.2f}%")
    print(f"Total Position Changes:  {total_trades:>8}")
    print(f"Days Locked Down (Brake):{days_locked_down:>8}")
    print(f"Number of Changepoints Detected:{total_changepoints:>8}")
    print("="*40 + "\n")

def run_strategy(data, chosen_a, chosen_b, long_window, z_entry):
    df = data.copy()
    spread, beta = calculate_spread(long_window, stock_a=chosen_a, stock_b=chosen_b, data=df)
    half_life = calculate_halflife(spread)
    if pd.isna(half_life) or half_life < 2:
        return None
    short_window_size=(np.floor(half_life)).astype(int)
    print("window is", short_window_size)
    df["z_score"] = z_scores(spread=spread, window=short_window_size)
    print("z scores between", df["z_score"].min(), " and ", df["z_score"].max())
    df = bayesian_cp(df, df["z_score"], chosen_a, chosen_b, long_window)
    df = generate_signals_and_returns(df, z_entry=z_entry)
    df = calculate_returns(chosen_a, chosen_b, beta, data=df)
    return df
####

def execute(combos, training_data, trading_data):
    #find pair to trade
    pvaldf = engle_granger(combos=combos, data=training_data)
    chosen_a, chosen_b = pvaldf.iloc[0, 0:2]
    print(f"Selected Pair: {chosen_a} , {chosen_b}")

    #tune params based on this pair
    print("Tuning parameters on training data")
    tuning_results = parameter_tuning(training_data, chosen_a, chosen_b)

    if tuning_results.empty:
        print("Parameter tuning failed.")
        return None

    plot_optimization_surface(tuning_results, metric="Calmar")
    try:
        best_window = int(input("Enter the chosen window: "))
        best_z = float(input("Enter the chosen z score threshold: "))
    except ValueError:
        print("Invalid input")
        return None
    
    out_of_sample_df = run_strategy(trading_data, chosen_a, chosen_b, best_window, best_z)
    
    if out_of_sample_df is None:
        print("Failed")
        return None
        
    out_of_sample_df["cumulative_return"].plot(
        figsize=(10, 5), 
        title=f"Return: {chosen_a} & {chosen_b}"
    )
    plt.show()
    print_strategy_summary(out_of_sample_df)
    
    return out_of_sample_df

execute(combos, training_data, trading_data)

