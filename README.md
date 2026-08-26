# Pairs Trading Strategy with Bayesian ChangePoint Detection
## Overview
* Statistical arbitrage trading algorithm designed to trade highly cointegrated equities.
* The model uses dynamic hedge ratios (Rolling OLS), half-life-derived Z-scores, and Bayesian Online Changepoint Detection (BOCD) to manage risk and adapt to changing market regimes.

## Methodology
* Cointegration Screening (Engle-Granger): Evaluates a selection of stocks to find the most statistically robust pairs over a given training period.

* Dynamic Hedge Ratios: Utilizes a Rolling Ordinary Least Squares (OLS) regression to calculate a dynamic beta, ensuring the spread remains stationary even as market volatility shifts.

* Adaptive Signal Generation: Derives the Z-score lookback window dynamically based on the spread's calculated Ornstein-Uhlenbeck half-life.

* Bayesian Changepoint Detection (BOCD): Acts as an emergency brake if a change point is detected to minimise loss. Brake lifted when cointegration reconfirmed. 

## Tech Stack
* Language: Python
* Data Sources: yfinance
* Libraries: statsmodels, numpy, pandas, bocd, itertools
* Visualization: matplotlib, seaborn

## Out-of-Sample Performance
The model was trained on historical data (2018–2019) and evaluated strictly out-of-sample (Jan 2020 – Jan 2023), accounting for a transaction cost of 0.05%. The following was achieved with a window of 22 days and a z-score threshold of 0.9. 
      
* Total Cumulative Return:    22.88%
* Sharpe Ratio:                1.12
* Maximum Drawdown:           -4.69%
* Total Position Changes:       196
* Days Locked Down (Brake):     128
* Number of Changepoints Detected:      19

<img width="865" height="448" alt="Screenshot 2026-08-26 at 15 15 00" src="https://github.com/user-attachments/assets/04e0ac21-ebdd-4b34-b13a-7f75d1915881" />
<img width="1091" height="636" alt="image" src="https://github.com/user-attachments/assets/9b13039f-c5ff-41c3-bc41-eaa6990e1fcb" />

## Next Steps
While the current architecture successfully proves the core statistical concepts, the next phase of development focuses on mitigating parameter decay:
Walk-Forward Optimization: Implementing a rolling training/trading window to allow the model to dynamically update the OLS window and Z-score entry thresholds over time.
Multi-Pair Rotation: Expanding the Walk-Forward loop to dynamically drop decoupled pairs and select new, highly cointegrated pairs if cointegration status changes over time. 
Automatic Max Detection: Currently optimal parameters are user input based on the heatmap plot. Possibly could use Gaussian smoothing filters to remove user input while maintaining stability of parameters. 
