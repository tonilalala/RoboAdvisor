# RoboAdvisor

### Overview
* Portfolio Construction: Portfolio_Construction.py, Portfolio_Construction.ipynb
* Portfolio Evaluation: Portfolio_Evaluation.py
* Code to be released: Risk Analysis Part  

### Introduction 
The purpose of this project is to establish a Robot-Advisor for investors. Our Robo Advisor provides three different portfolios (low, medium and high) for people with di↵erent risk appetite, which will be determined by the risk score in the IPS questionnaire. Each portfolio associated with different risk levels is constructed under different return and volatility requirements.
In this report, we summarized the performance of our portfolios with low, medium and high risk level. We divided our capital into both US market and Canadian market. Portfolio assets and weights are selected based on investors risk appetite and will be rebalanced every 6 months during the 5-year investment horizon. We use both nominal Mean-Variance Optimization (nominal MVO) and Risk Parity models to construct our portfolio. We will invest in Equity, Fixed Income, Real Estate and Commodity for high risk portfolio. For low and medium risk portfolio, we will invest in Equity, Fixed Income, Real Estate but not in Commodity, which is relatively more volatile. At last, each of these portfolios’ performance will be compared with a specific benchmark.
Different performance and risk metrics of the portfolios are illustrated in the report including quarterly P&L, Value at Risk (VaR), Expected Shortfall, Sharpe Ratio, Risk Contribution, and some other risk measures. Furthermore, we also implemented scenario tests and stress test, which evaluates the performance of the portfolio under di↵erent economic scenarios.

