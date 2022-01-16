# Portfolio Construction Functions
# from __future__ import division
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv,pinv
from scipy.optimize import minimize

#################### Helper Functions ##################
def ObjectiveFunc(x, cov_matrix, mean_v,lbd):
    """
    x is the weight vector
    lambda = 0.5
    """
    obj=np.dot(x,np.dot(cov_matrix,x))-sum(np.multiply(mean_v,x)*lbd)
    return(obj)

def nominal_mvo(mu,Q,n,lbd):
    cons = ({'type': 'eq',
                    'fun': lambda x: 1-sum(x)}, {'type':'ineq', 
                                                 'fun': lambda x: x})
    x = np.array([1/n]*n)
    res = minimize(ObjectiveFunc, x, args=(Q, mu,lbd),
                   method='SLSQP', constraints=cons)
    return res.x 

def ObjectiveFunc_SR(x, cov_matrix, mean_v):
    # objective function: sharpe ratio
    obj=sum(np.multiply(mean_v,x))/(np.dot(x,np.dot(cov_matrix,x)))**(0.5)
    return(obj)

def nominal_mvo2(mu,Q,n):
    cons = ({'type': 'eq',
                    'fun': lambda x: 1-sum(x)}, 
            {'type':'ineq', 'fun': lambda x: x})
    x = np.array([1/n]*n)
    res = minimize(ObjectiveFunc_SR, x, args=(Q, mu), method='SLSQP', constraints=cons)

    return res.x 


 # risk budgeting optimization
def calculate_portfolio_var(w,V):
    # function that calculates portfolio risk
    w = np.matrix(w)
    return (w*V*w.T)[0,0]

def calculate_risk_contribution(w,V):
    # function that calculates asset contribution to total risk
    w = np.matrix(w)
    sigma = np.sqrt(calculate_portfolio_var(w,V))
    # Marginal Risk Contribution
    MRC = V*w.T
    # Risk Contribution
    RC = np.multiply(MRC,w.T)/sigma
    return RC

def risk_budget_objective(x,pars):
    # calculate portfolio risk
    V = pars[0]# covariance table
    x_t = pars[1] # risk target in percent of portfolio risk
    sig_p =  np.sqrt(calculate_portfolio_var(x,V)) # portfolio sigma
    risk_target = np.asmatrix(np.multiply(sig_p,x_t))
    asset_RC = calculate_risk_contribution(x,V)
    J = sum(np.square(asset_RC-risk_target.T))[0,0]*10000 # sum of squared error
    return J

def total_weight_constraint(x):
    return np.sum(x)-1.0

def long_only_constraint(x):
    return x

########################################################
def Portfolio_Construction(df_data,Method,Risk):
    """
    df_data: dataset
    Method: "MVO_0": use mvo for optimization, lambda = 0
            "MVO_0.5": use mvo for optimization, lambda = 0.5
            "RiskParity": use risk parity for portfolio construction
    Risk: Low, Mid, High
    """
    investment_date = pd.to_datetime('2014-04-01')
    start_date = pd.to_datetime('2014-09-01')
    end_date = pd.to_datetime('2019-06-01')
    i = 0
    FX0 = FX_data.loc[FX_data['Dates']==investment_date,'PX_LAST']

    if Risk == "Low" or Risk == "Mid":
        FX0_vector=np.append([float(FX0)]*8,[1]*5)
    elif Risk == 'High':
        FX0_vector=np.append([float(FX0)]*14,[1]*5)

    portfolio0 = 200000-1000
    additional_val = 20000
    portfolio_value = np.array([investment_date, portfolio0])
    portfolio_investment = np.array([investment_date, -portfolio0])
    portfolio_returns = np.array([investment_date,1])
    pl = np.array([investment_date, 0])
    portfolio_PnL = np.array([investment_date, 0])
    index = [0]
    df_data1 = df_data
    df_data1 = df_data1[df_data1['Dates'] >= investment_date]
    df_data2 = df_data.copy()
    
    df_data2 = df_data2[df_data2['Dates'] < investment_date]
    
    prev_price = df_data2.iloc[-1].to_numpy()
    df_data2.drop(columns = 'Dates', axis=1, inplace=True)
    df_data2 = df_data2.pct_change()
    df_data2=df_data2.replace([np.inf, -np.inf], np.nan)
    df_data2 = df_data2.fillna(0)

    mu1 = df_data2.mean().to_numpy()
    Q1 = df_data2.cov().to_numpy()
    n = len(df_data2.columns)

    if Method == "MVO_0":
        w = nominal_mvo(mu1,Q1,n,0)
    elif Method == "MVO_0.5":
        w = nominal_mvo(mu1,Q1,n,0.5)
    elif Method == "MVO_1":
        w = nominal_mvo(mu1,Q1,n,1)
    elif Method == "MVO_1.5":
        w = nominal_mvo(mu1,Q1,n,1.5)
    elif Method == "MVO_2":
        w = nominal_mvo(mu1,Q1,n,2)
    elif Method == "RiskParity":
        w0 = [1/n]*n
        x_t = [1/n]*n # your risk budget percent of total portfolio risk (equal risk)
        cons = ({'type': 'eq', 'fun': total_weight_constraint})
        res= minimize(risk_budget_objective, w0, args=[Q1,x_t], 
                      method='SLSQP',constraints=cons, options={'disp': True})
        w = res.x
    else:
        print("Please enter correct method.")
        assert False
    weights_initial = w

    prev_investment = 0
    w_prev = [0]*len(w)
    prev_portfolio_holdings = w*portfolio0/prev_price[1:]
    TransactionCost = 0
    portfolio_holdings_list = []
    while start_date <= end_date:
        # Number of holdings Vector
        FX1 = FX_data.iloc[i,1]
        if Risk == "Low" or Risk == "Mid":
            FX1_vector=np.append([float(FX1)]*8,[1]*5)
        elif Risk == 'High':
            FX1_vector=np.append([float(FX1)]*14,[1]*5)
        portfolio_holdings = w*portfolio0/(prev_price[1:]/FX1_vector)
        portfolio_holdings_list = [portfolio_holdings_list,portfolio_holdings]

        while df_data1.iloc[i]['Dates'] <= start_date:
            cur_price = df_data1.iloc[i].to_numpy()

            # PnL from Adjusting portfolio allocation
            PnL = (prev_portfolio_holdings - portfolio_holdings).dot(
                cur_price[1:]/FX1_vector-prev_price[1:]/FX0_vector)-TransactionCost
        
            cur_PnL = np.array([df_data1.iloc[i]['Dates'],PnL])
            portfolio_PnL = np.vstack((portfolio_PnL, cur_PnL))
            cur_value = np.array([df_data1.iloc[i]['Dates'], sum(
                np.multiply(cur_price[1:], portfolio_holdings))-TransactionCost])
            
            portfolio_value = np.vstack((portfolio_value, cur_value))

            cur_investment = np.array([df_data1.iloc[i]['Dates'], min(0,prev_investment)])
            prev_investment = 0
            portfolio_investment =  np.vstack((portfolio_investment,cur_investment))

            cur_returns = np.array([df_data1.iloc[i]['Dates'],sum(
                np.multiply(cur_price[1:], w/prev_price[1:]))])
            portfolio_returns =  np.vstack((portfolio_returns,cur_returns))

            prev_price = cur_price


            cur_pl = np.array([df_data1.iloc[i]['Dates'], 
                               portfolio_value[i + 1, 1] - portfolio_value[i, 1]])
            pl = np.vstack((pl, cur_pl))

            i = i + 1
            prev_portfolio_holdings = portfolio_holdings
            FX0_vector = FX1_vector

        index.append(i)

        df_data3 = df_data.copy()
        df_data3 = df_data3[investment_date < df_data3['Dates']]
        df_data3 = df_data3[df_data3['Dates'] < start_date]

        df_data3.drop(columns = 'Dates', axis=1, inplace=True)
        df_data3 = df_data3.pct_change()
        df_data3 = df_data3.fillna(0)

        mu1 = df_data3.mean().to_numpy()
        Q1 = df_data3.cov().to_numpy()
        n = len(df_data3.columns)

        current_price1 = df_data.iloc[i]
        current_price1 = current_price1.values.tolist()
        
        if Method == "MVO_0":
            w = nominal_mvo(mu1,Q1,n,0)
        elif Method == "MVO_0.5":
            w = nominal_mvo(mu1,Q1,n,0.5) 
        elif Method == "MVO_1":
            w = nominal_mvo(mu1,Q1,n,1)
        elif Method == "MVO_1.5":
            w = nominal_mvo(mu1,Q1,n,1.5)
        elif Method == "MVO_2":
            w = nominal_mvo(mu1,Q1,n,2)
        elif Method == "RiskParity":
            w0 = [1/n]*n
            x_t = [1/n]*n # your risk budget percent of total portfolio risk (equal risk)

            cons = ({'type': 'eq', 'fun': total_weight_constraint})
            res= minimize(risk_budget_objective, w0, args=[Q1,x_t], 
                          method='SLSQP',constraints=cons, options={'disp': True})
            w = res.x
        else:
            print("Please enter correct method.")
            assert False
        w_prev = w
        print(w)
        portfolio0 = portfolio0 + additional_val
        prev_investment = -additional_val
        investment_date = start_date
        start_date = start_date + pd.DateOffset(months = 6)
        prev_portfolio_holdings = portfolio_holdings
        TransactionCost = abs(np.array(w_prev)-np.array(w)).dot(
            cur_price[1:])*portfolio0/sum(cur_price[1:])*0.0117 
        # 117bps of transaction amount for transaction cost
    return portfolio_value,portfolio_investment,pl,portfolio_returns,portfolio_PnL,weights_initial,w,portfolio_holdings_list

