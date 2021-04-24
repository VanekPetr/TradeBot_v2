"""
Created on Wed Nov 11 15:50:42 2020

@author: Petr Vanek
"""
import numpy as np

from dataAnalyser import analyseData, getStat, finalStat
from dataGraph import plotInteractive, plotOptimization
from MST import MinimumSpanningTree
from Clustering import Cluster, pickCluster
from ScenarioGeneration import MC, BOOT
from CVaRtargets import targetsCVaR
from CVaRmodel import modelCVaR

from pandas_datareader import data


# WHAT TIME PERIOD DO WE WANT WORK WITH?
#------------------------------------------------------------------
startDate = "2015-09-23"
endDate = "2019-09-22"
#thesis dates:  startDate ="2016-07-25" & endDate = "2020-04-30"

# TICKERS OF ETFs WE ARE GOING TO WORK WITH
#------------------------------------------------------------------

tickers = ["ANGL","ASHR","BIV","BKLN","BNDX","BOND","BRF","CGW","CMBS",
           "CMF","CORP","CSM","CWB","DBA","DBB","DBO","DBS","DBV","DES","DGL",
           "DGRW","DIA","DLS","DOL","DON","DSI","DXGE","DXJ","EBND","ECH","EDEN",
           "EEM","EFV","EIDO","EIRL","ENZL","EPHE","EWD","EWG","EWH","EWI","EWL",
           "EWM","EWN","EWQ","EWS","EWT","EWU","EWW","EWY","EZA","FBT","FCG",
           "FCOM","FDD","FDL","FEU","FEX","FLRN","FPX","FTA","FTCS","FTSM","FXA",
           "FXB","FXC","FXE","FXF","FXR","FXY","FXZ","GBF","GNMA","GREK","GSY",
           "HDV","HEDJ","HEFA","HYD","HYEM","IAU","IBND","IDLV","IGE","IGN","IGV",
           "IHI","INDA","IOO","IPE","IPFF","IQDF","ISTB","ITA","ITB","ITM","IUSV",
           "IVOO","IVOV","IWC","IWN","IWO","IWY","IXG","IXN","IYE","IYY","IYZ",
           "JKD","JKE","JKG","JPXN","KBWP","KOL","KRE","KXI","LTPZ","MCHI","MDYG",
           "MDYV","MGV","MLPA","MOAT","MOO","MTUM","OIH","PALL","PCY","PDP","PEY",
           "PFXF","PHB","PHO","PJP","PKW","PRFZ","PSCC","PSCT","PSCU","PSK","PSL",
           "PUI","PWB","PWV","PWZ","QDF","QUAL","RDIV","REM","REZ","RFG","RING",
           "RSX","RTH","RWJ","RWL","RWX","RXI","RYF","SCHC","SCHE","SCJ","SDIV",
           "SDOG","SGDM","SGOL","SHM","SILJ","SIZE","SLQD","SLY","SLYG","SMLV",
           "SNLN","SOCL","SPHQ","SPYG","TAN","TDIV","TDTT","THD","TIP","TOK","TUR",
           "UGA","URA","URTH","USDU","VBK","VCLT","VEA","VLUE","VNM","VOE","VONE",
           "VONG","VONV","VOT","VXF","XBI","XES","XHS","XLE","XLG","XLI","XLK",
           "XLP","XLU","XLV","XLY","XME","XPH","XRT","XSD","XTN","ZROZ"]

#tickers = ['BTC-USD']
"""
    ----------------------------------------------------------------------
    DATA ANALYTICS AND VISUALISATION 
    ----------------------------------------------------------------------
"""
# DOWNLOAD THE DATA FROM YAHOO DATABASE
#------------------------------------------------------------------
dailyPrices = data.DataReader(tickers, 'yahoo', startDate, endDate)
dailyPrices = dailyPrices["Adj Close"]
    

# ANALYSE THE DATA AND GET WEEKLY RETURNS  
#------------------------------------------------------------------
dataStat, weeklyReturns = analyseData(data = dailyPrices,
                                      startDate = startDate,
                                      endDate = endDate) 


# PLOT INTERACTIVE GRAPH
#------------------------------------------------------------------
plotInteractive(data = dataStat,
                start = startDate,
                end = endDate,
                ML = None, MLsubset = None)



"""
    ----------------------------------------------------------------------
    MACHINE LEARNING AND ADVANCED STATISTICAL METHODS
    ----------------------------------------------------------------------
""" 
# DIVIDE DATASET INTO TRAINING AND TESTING PART?
#------------------------------------------------------------------
divide = True

# IF WE DIVIDE DATASET
if divide != False:
    # ONE HALF OF THE DATA, BREAKPOINT IN TRAINING AND TESTING DATASET
    breakPoint = int(np.floor(len(weeklyReturns.index)/2))
    # DEFINITION OF TRAINING AND TESTING DATASETS
    trainDataset = weeklyReturns.iloc[0:breakPoint,:] 
    testDataset = weeklyReturns.iloc[breakPoint:,:] 
    
    dataPlot = getStat(data = trainDataset)
    endTrainDate = str(trainDataset.index.date[-1])
    startTestDate = str(testDataset.index.date[0])
    lenTest = len(testDataset.index)
else: 
    trainDataset = weeklyReturns
    dataPlot = dataStat
    lenTest = 0
    

# RUN THE MINIMUM SPANNING TREE METHOD
#------------------------------------------------------------------
nMST = 3                        # Select how many times run the MST method   
subsetMST_df = trainDataset
for i in range(nMST):
    subsetMST, subsetMST_df, corrMST_avg, PDI_MST = MinimumSpanningTree(subsetMST_df)
    
# PLOT
plotInteractive(data = dataPlot,
                ML = "MST", 
                MLsubset = subsetMST,
                start = startDate,
                end = endTrainDate)
    
 
# RUN THE CLUSTERING METHOD 
#------------------------------------------------------------------
clusters = Cluster(trainDataset,
                   nClusters = 3,
                   dendogram = False)
    
# SELECT ASSETS
subsetCLUST, subsetCLUST_df = pickCluster(data = trainDataset,
                                          stat = dataPlot,
                                          ML = clusters,
                                          nAssets = 10) #Number of assets selected from each cluster

# PLOT
plotInteractive(data = dataPlot,
                ML = "Clustering", 
                MLsubset = clusters,
                start = startDate,
                end = endTrainDate)
    
    

"""
    ----------------------------------------------------------------------
    SCENARIO GENERATION
    ----------------------------------------------------------------------
""" 

# THE BOOTSTRAPPING SCENARIO GENERATION 
#------------------------------------------------------------------
# FOR THE MST METHOD
BOOT_sim_MST = BOOT(data = weeklyReturns[subsetMST],    #subsetMST or subsetCLUST
                nSim = 250,                             #number of scenarios per period
                N_test = lenTest) 


# FOR THE CLUSTERING METHOD
BOOT_sim_CLUST = BOOT(data = weeklyReturns[subsetCLUST],#subsetMST or subsetCLUST
                      nSim = 250,
                      N_test = lenTest) 

# THE MONTE CARLO SCENARIO GENERATION 
#------------------------------------------------------------------
# FOR THE MST METHOD
MC_sim_MST = MC(data = subsetMST_df,                    #subsetMST_df or subsetCLUST_df
                nSim = 250,
                N_test = lenTest)    



# FOR THE CLUSTERING METHOD  
MC_sim_CLUST = MC(data = subsetCLUST_df,                #subsetMST_df or subsetCLUST_df
                  nSim = 250,
                  N_test = lenTest)  


"""
    ----------------------------------------------------------------------
    MATHEMATICAL OPTIMIZATION
    ----------------------------------------------------------------------
""" 
# TARGETS GENERATION
#------------------------------------------------------------------
targets, benchmarkPortVal = targetsCVaR(start_date = startDate,
                                        end_date = endDate,
                                        test_date = startTestDate,
                                        benchmark = ["URTH"],   #MSCI World benchmark
                                        test_index = testDataset.index.date,
                                        budget = 100,
                                        cvar_alpha=0.05) 


# MATHEMATICAL MODELING
#------------------------------------------------------------------
portAllocation, portValue, portCVaR = modelCVaR(testRet = testDataset[subsetMST],
                                                scen = MC_sim_MST,  #Scenarios
                                                targets = targets,  #Target
                                                budget = 100,
                                                cvar_alpha = 0.05,
                                                trans_cost = 0.001,
                                                max_weight = 0.33)

# PLOTTING
#------------------------------------------------------------------
plotOptimization(performance = portValue.copy(),
                 performanceBenchmark = benchmarkPortVal.copy(),
                 composition = portAllocation,
                 names = subsetMST)



# STATISTICS
#------------------------------------------------------------------
finalStat(portValue) 
finalStat(benchmarkPortVal)
