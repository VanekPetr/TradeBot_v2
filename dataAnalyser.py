#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 17:59:42 2020

@author: Petr Vanek
"""

import pandas as pd
import numpy as np
import math


# Function computing the geometric mean of annual returns 
#----------------------------------------------------------------------     
def meanRetAn(data):             
    Result = 1
    
    for i in range(len(data.index)):
        Result *= (1+data.iloc[i,:])
        
    Result = Result**(1/float(len(data.index)/52))-1
     
    return(Result)
    
"""
    ----------------------------------------------------------------------
    Data Analytics and Visualisation: ANALYSE THE DATA AND GET WEEKLY RETURNS  
    ---------------------------------------------------------------------- 
"""  
def analyseData(data, startDate, endDate):
   
    # Modify the pandas table (get returns for a given time period)
    #---------------------------------------------------------------------- 
    
    # DEFINE IF WE WORK WITH ISIN CODES OR NAMES OF MUTUAL FUNDS
    workPrices = data
    
    # MODIFY THE DATA
    pricesWed = workPrices[workPrices.index.weekday==2]     ## 1, Only wednesdays

    startdate = startDate                                   ## 2, Subset of dates                               
    enddate = endDate
    
    pricesSub = pricesWed[pricesWed.index < enddate]        #define time interval
    pricesSub = pricesSub[pricesSub.index > startdate]
    
    weeklyPrices = pricesSub.dropna(axis=1)                 #delete NaN columns
    
    # GET WEEKLY RETURNS
    weeklyReturns = weeklyPrices.pct_change()                  
    weeklyReturns = weeklyReturns.drop(weeklyReturns.index[:1])     #drop first NaN row
    
    
    # GET MONTHLY RETURNS
    numMonths = math.floor((len(weeklyPrices.index)-1)/4)      
    monthlyPrices = pd.DataFrame(columns=weeklyPrices.columns)
    for i in range(numMonths+1):
        oneRow = weeklyPrices.iloc[0+4*i,:]
        monthlyPrices = monthlyPrices.append(oneRow)
        
    monthlyReturns = monthlyPrices.pct_change()
    monthlyReturns = monthlyReturns.drop(monthlyReturns.index[:1])
    
    """
    # 5, Get Annual returns
    N_results = len(monthlyPrices.index)-12
    annualReturns = pd.DataFrame(columns=monthlyPrices.columns)
    for i in range(N_results):
        annualReturns = annualReturns.append((monthlyPrices.iloc[i+12,:]
        - monthlyPrices.iloc[i,:])/monthlyPrices.iloc[i,:], ignore_index=True)
    
    # 5, Get Annual returns
    N_results = int(np.floor((len(monthlyPrices.index)-1)/12))
    zbyt = (len(monthlyPrices.index)-1)%12
    annualReturns = pd.DataFrame(columns=monthlyPrices.columns)   
    for i in range(N_results):
        annualReturns = annualReturns.append((monthlyPrices.iloc[zbyt + 12 + 12*i,:]
        - monthlyPrices.iloc[zbyt + i*12,:])/monthlyPrices.iloc[zbyt + i*12,:],
            ignore_index=True)
    """
    
    # Analysis of the data
    #---------------------------------------------------------------------- 
    
    # TABLE WITH AVG RET AND STD OF RET
    mu_ga = meanRetAn(weeklyReturns)                   #anual geometric mean
    stdev_a = weeklyReturns.std(axis=0) * np.sqrt(52)   #standard deviation of Annual Returns
    #stdev_a = annualReturns.std(axis=0)        

    statDf = pd.concat([mu_ga,stdev_a], axis=1)         #table
    statName = ["Average Annual Returns","Standard Deviation of Returns"]
    statDf.columns = statName                           #add names
    
    # COMPUTE SHARPE RATIO AND ADD IT INTO THE TABLE
    sharpe = statDf.loc[:,"Average Annual Returns"]/statDf.loc[:,"Standard Deviation of Returns"]
    statDf = pd.concat([statDf,sharpe], axis=1)         #add sharpe ratio into the table

    statName = ["Average Annual Returns","Standard Deviation of Returns",
                "Sharpe Ratio"]
    statDf.columns = statName 
    
    # ADD NAMES INTO THE TABLE
    statDf["Name"] = statDf.index
    
    return(statDf, weeklyReturns)

"""
    ----------------------------------------------------------------------
    Machine Learning and Advanced Statistical Methods: GET STATISTICS BASED ON WEEKLY RETURNS 
    ---------------------------------------------------------------------- 
"""      
def getStat(data):
    # TABLE WITH AVG RET AND STD OF RET
    mu_ga = meanRetAn(data)                   #anual geometric mean
    stdev_a = data.std(axis=0) * np.sqrt(52)   #standard deviation of Annual Returns
    #stdev_a = annualReturns.std(axis=0)        

    statDf = pd.concat([mu_ga,stdev_a], axis=1)         #table
    statName = ["Average Annual Returns","Standard Deviation of Returns"]
    statDf.columns = statName                           #add names
    
    # COMPUTE SHARPE RATIO AND ADD IT INTO THE TABLE
    sharpe = statDf.loc[:,"Average Annual Returns"]/statDf.loc[:,"Standard Deviation of Returns"]
    statDf = pd.concat([statDf,sharpe], axis=1)         #add sharpe ratio into the table
    statName = ["Average Annual Returns","Standard Deviation of Returns", "Sharpe Ratio"]
    statDf.columns = statName
    
    # ADD NAMES INTO THE TABLE
    statDf["Name"] = data.columns 
    
    return(statDf)    


"""
    ----------------------------------------------------------------------
    Mathematical Optimization: GET PERFORMANCE STATISTICS FOR OUR RESULTS
    ---------------------------------------------------------------------- 
"""     
def finalStat(data):
    # TABLE WITH AVG RET AND STD OF RET
    data = data.pct_change()                  
    data = data.drop(data.index[:1])
    
    mu_ga = meanRetAn(data)                   #anual geometric mean
    stdev_a = data.std(axis=0) * np.sqrt(52)   #standard deviation of Annual Returns
    #stdev_a = annualReturns.std(axis=0)        

    statDf = pd.concat([mu_ga,stdev_a], axis=1)         #table
    statName = ["Average Annual Returns","Standard Deviation of Returns"]
    statDf.columns = statName                           #add names
    
    # COMPUTE SHARPE RATIO AND ADD IT INTO THE TABLE
    sharpe = statDf.loc[:,"Average Annual Returns"]/statDf.loc[:,"Standard Deviation of Returns"]
    statDf = pd.concat([statDf,sharpe], axis=1)         #add sharpe ratio into the table
    statName = ["Avg An Ret","Std Dev of Ret", "Sharpe R"]
    statDf.columns = statName
    
    return(statDf)

"""
    ----------------------------------------------------------------------
    CVaR targets file: GET PERFORMANCE STATISTICS FOR OUR RESULTS
    ---------------------------------------------------------------------- 
"""   
def getWeeklyRet(data):
    # DEFINE IF WE WORK WITH ISIN CODES OR NAMES OF MUTUAL FUNDS
    workPrices = data
    # MODIFY THE DATA
    pricesWed = workPrices[workPrices.index.weekday==2]     ## 1, Only wednesdays

    #Get weekly returns
    weeklyReturns = pricesWed.pct_change()                  
    weeklyReturns = weeklyReturns.drop(weeklyReturns.index[:1])     #drop first NaN row

    return weeklyReturns
