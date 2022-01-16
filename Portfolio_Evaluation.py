# Portfolio Evaluation Functions
import pandas as pd
from scipy.stats import norm
from tabulate import tabulate
import cvxopt

def AnnualizedReturns(Returns,Num_periods):
    """
    Returns: A column from dataframe with returns 
    Num_periods: An integer
    """
    result = Returns.mean(skipna = True)*Num_periods
    return(round(result,3))

def AnnualizedVolatility(Returns,Num_periods):
    """
    Returns: A column from dataframe with returns 
    Num_periods: An integer
    """
    result = Returns.std(skipna = True)*(Num_periods**(1/2))
    return(round(result,3))

def SharpeRatio(Returns,Num_periods):
    """
    Returns: A column from dataframe with returns 
    Num_periods: An integer
    """
    result = AnnualizedReturns(Returns,Num_periods)/AnnualizedVolatility(Returns,Num_periods)
    return(round(result,3))

def InformationRatio(Returns,Benchmark,Num_periods):
    """
    Returns: A column from dataframe with portfolio returns 
    Benchmark: A column from dataframe with benchmark returns 
    Num_periods: An integer
    """
    result = AnnualizedReturns(Returns-Benchmark,Num_periods)/AnnualizedVolatility(Returns-Benchmark,Num_periods)
    return(round(result,3))

def VaR_normal(Returns,Num_periods,alpha):
    """
    Returns: A column from dataframe with portfolio returns 
    Num_periods: An integer
    percentile: probability of losing at least VaR amount money
    """
    percentile=1-alpha
    result=abs(norm.ppf(percentile,Returns.mean(skipna=True)*Num_periods,Returns.std(skipna=True)*(Num_periods**(1/2))))
    return(round(result,3))

def VaR_historical(Returns,Num_periods,alpha):
    """
    Returns: A column from dataframe with portfolio returns 
    Num_periods: An integer
    percentile: probability of losing at least VaR amount money
    """
    percentile=1-alpha
    result = abs(Returns.quantile(percentile)*Num_periods)
    return(round(result,3))


def CVaR(Returns,Num_periods,alpha):
    """
    Returns: A column from dataframe with portfolio returns 
    Num_periods: An integer
    percentile: probability of losing at least VaR amount money
    """
    mu_h = Returns.mean(skipna=True)
    sig_h = Returns.std(skipna=True)
    CVaR_n = alpha**-1 * norm.pdf(norm.ppf(alpha))*sig_h - mu_h
    return(round(CVaR_n,3))

def week_to_quarter(df_data):
    start_date = get_closest_quarter(df_data['Dates'].iloc[0])
    end_date = df_data['Dates'].iloc[-1]
    near_date = nearest(df_data['Dates'], start_date)
    new_row = df_data[df_data['Dates'] == near_date]
    new_row['Dates'].iloc[0] = start_date
    new_df = new_row.copy()
    start_date = start_date + pd.DateOffset(months = 3)
    while start_date <= end_date:
        near_date = nearest(df_data['Dates'], start_date)
        new_row = df_data[df_data['Dates'] == near_date]
        new_row['Dates'].iloc[0] = start_date
        new_df = new_df.append(new_row)
        start_date = start_date + pd.DateOffset(months = 3)   
    return new_df

################### Other Useful Functions ###################
def get_closest_quarter(target):
    # candidate list, nicely enough none of these 
    # are in February, so the month lengths are fixed
    candidates = [
        pd.Timestamp(year=target.year, month=1, day=1),
        pd.Timestamp(year=target.year, month=4, day=1),
        pd.Timestamp(year=target.year, month=7, day=1),
        pd.Timestamp(year=target.year, month=10, day=1),
        pd.Timestamp(year=target.year + 1, month=1, day=1),
    ]
    # take the minimum according to the absolute distance to
    # the target date.
    return min(candidates, key=lambda d: abs(target - d))

def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))