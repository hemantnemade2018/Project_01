#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""


#use 3 factor model FRENCH_FIMA_FACTOR csv from http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html#Research

import os             
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab
from sklearn.neighbors import KernelDensity
import scipy
from scipy.optimize import minimize
from sklearn import linear_model
from scipy.stats import kurtosis,skew,norm
import pandas as pd
import csv
import yfinance as yf
import time

#lets try this changes for 


# OPTIMAL WEIGHTS
def arg_w(rho, lamb, Q, wp, beta_im_ ,beta_T):        
    def constrain1(w):                        
        return np.dot(beta_im_,w)-beta_T  
    
#constrain defined according investment strategy 

    def constrain2(w):
        return np.sum(w)-1         
    
#constrain defined according investment strategy 

    cons = [{'type':'eq', 'fun': constrain1},
            {'type':'eq', 'fun': constrain2}]
# set bound 200% to -200% on portfolio
    
    bound_constrain = scipy.optimize.Bounds(-2.0, 2.0, keep_feasible = True)  
    

# maximazation of portfolio(we converted equation to negative so minization of this equation is maximization of portfolio
    def f(w):
        return -rho.dot(w) + lamb*(w-wp).dot(Q.dot(w-wp))                         

    w0 = np.array([1/12]*12)
# call quadratic equation here we used SLSQP METHOD
    res = minimize(f, w0, method='SLSQP', bounds=bound_constrain, constraints=cons, tol=1e-9) 

                  
    return res.x

#3FACTOR MODEL AND PORTFOLIO RETURN 
def get_omega(return_r, factor_r, return_v, factor_v, lamb_, beta_tm_, wp_):
    
#3 FACTOR MODEL INPUT FOR ANALYSIS    
    rf = np.asarray(factor_r['RF'])
    rM_rf = np.asarray(factor_r['Mkt-RF'])
    rSMB = np.asarray(factor_r['SMB'])
    rHML = np.asarray(factor_r['HML'])
    
#S&P ETF RETURN OVER PERIOD   
    SPY = np.asarray(return_r['SPY'])
    
#CONVERTED ARRAY INTO ASARRAY 
    ri = np.asarray(return_r)

#CALCULATE VAR OF MARKET   
    var_market = np.var(SPY,ddof=1)
    
#TEMP ARRAY FOR BETA
    beta_im = np.array([0.0]*12)
    
#CALCULATED BETA FOR PORTFOLIO
    for i in range (12):
        temp = np.cov(ri[:,i],SPY,ddof=1)
        beta_im[i] = temp[0,1] / var_market
    
    Ri = ri - rf.reshape(-1,1)
    
#CREATE 3 FACTOR INPUT ARRAY 

    f = np.array([rM_rf, rSMB, rHML])
    
#USED Transpose For Data Manipulation 
    F = f.T

#3 FACTOR MODEL LINAR REGRESSION

    lr = linear_model.LinearRegression().fit(F, Ri)
    
#GET ALPHA
    ALPHA = lr.intercept_
#GET Coefficient 
    B = lr.coef_
# get rho of short period
    ft = f[:,-1]
    rho_r = ALPHA + B.dot(ft) + rf[-1]
    
# --------------------------VARINACE 3 FACTOR MODEL------------------------------------------
    rf_v = np.asarray(factor_v['RF'])
    rM_rf_v = np.asarray(factor_v['Mkt-RF'])
    rSMB_v = np.asarray(factor_v['SMB'])
    rHML_v = np.asarray(factor_v['HML'])
    SPY_v = np.asarray(return_v['SPY'])
    
    ri_v = np.asarray(return_v)
    
    var_market_v = np.var(SPY_v,ddof=1)
    
#Beta for variance period 

    beta_im_v = np.array([0.0]*12)
    for i in range (12):
        temp_v = np.cov(ri_v[:,i],SPY_v,ddof=1)
        beta_im_v[i] = temp_v[0,1] / var_market_v
    
#variance calculation 
    Ri_v = ri_v - rf_v.reshape(-1,1)
    f_v = np.array([rM_rf_v, rSMB_v, rHML_v])

    F_v = f_v.T
    
    lr_v = linear_model.LinearRegression().fit(F_v, Ri_v)
#GET ALPHA
    ALPHA_v = lr_v.intercept_
    
#GET Coefficient
    B_v = lr_v.coef_
    
#CREATE INOUT FOR 3 FACTAR VARINACE
    eph_v = Ri_v.T - (ALPHA_v.reshape(-1,1) + B_v.dot(f_v))
    eph2_v = np.cov(eph_v,ddof=1)
    eph2_diag_v = np.diag(eph2_v)
    D_v = np.diag(eph2_diag_v)

    omega_f_v = np.cov(f_v,ddof=1)
#3 FACTOR MODEL   
    cov_Rt_v = B_v.dot(omega_f_v).dot(B_v.T) + D_v

    
    result = arg_w(rho_r, lamb_, cov_Rt_v, wp_, beta_im_v ,beta_tm_)
    
    return result

#RISK ANLYSIS ON DATA 
def RISK_ANALYIS(X,rf,confidenceLevel,position):
#PnL
    
    DAILY_CUM_RETURN=np.cumprod((X+1))
    ANNUAL_CUM_RETURN = (np.power(DAILY_CUM_RETURN.iloc[-1,0],1/len(X)))**250

#Daily Mean return(%):
    ANNUAL_ARTH_MEAN_RETURN=np.mean(X)*250
    
#Geomean is zero i
    ANNUAL_GEO_MEAN_RETURN=(np.power(DAILY_CUM_RETURN.iloc[-1,0],1/len(X))-1)*252
    
# Min Return
    ANNUAL_MIN_RETURN = np.min(X)*250
    
#MDD
    p_v =np.cumprod((X+1))*100
    p_v_extend = pd.DataFrame(np.append([p_v.iloc[0,0]]*9,p_v))
    Roll_Max = p_v_extend.rolling(window=10).max()
    
    TEN_DAY_DRAWDOWN = float(np.min(p_v_extend/Roll_Max-1)[0])
       
#volatility
    ANNUAL_VOLATILITY=np.std(X)*np.sqrt(250)
    
#Sharp ratio
    SHARP_RATIO_ANNUAL=(ANNUAL_ARTH_MEAN_RETURN-rf)/ANNUAL_VOLATILITY
    
    
#Skewness, Kurtosis
    annual_Kurt=kurtosis(X*250)
    annual_sk=skew(X*250)
    
#MVaR
    DAILY_KURTOSIS=kurtosis(X)
    DAILY_SKWNESS=skew(X)


    z=norm.ppf(1-confidenceLevel)
    t=z+((1/6)*(z**2-1)*DAILY_SKWNESS)+((1/24)*(z**3-3*z))*DAILY_KURTOSIS-((1/36)*(2*z**3-5*z)*(DAILY_SKWNESS**2))
    mVaR= position*(np.mean(X)+t*np.std(X))*np.sqrt(250)
    
    
    ALPHA=norm.ppf(1-confidenceLevel, np.mean(X), np.std(X))
    
#VAR    
    VaR= position*(ALPHA)
    ANNUAL_VaR=VaR*np.sqrt(250)
#CVAR
=
WHOLE[1].hist(ax=ax,bins=50)

