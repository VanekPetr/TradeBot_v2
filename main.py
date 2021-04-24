"""
Created on Wed Nov 11 15:50:42 2020

@author: Petr Vanek
"""
import numpy as np

from dataAnalyser import meanRetAn, finalStat
from dataGraph import plotInteractive, plotOptimization
from MST import MinimumSpanningTree
from Clustering import Cluster, pickCluster
from ScenarioGeneration import MC, BOOT
from CVaRtargets import targetsCVaR
from CVaRmodel import modelCVaR

from pandas_datareader import data
import pandas as pd
import plotly.express as px


class TradeBot(object):
    """
    Python class analysing financial products and based on machine learning algorithms and mathematical
    optimization suggesting optimal portfolio of assets.
    """

    def __init__(self, start, end, assets):
        # DOWNLOAD THE ADJUSTED DAILY PRICES FROM YAHOO DATABASE
        dailyPrices = data.DataReader(assets, 'yahoo', start, end)["Adj Close"]

        # GET WEEKLY RETURNS
        # Get prices only for Wednesdays and delete Nan columns
        pricesWed = dailyPrices[dailyPrices.index.weekday == 2].dropna(axis=1)
        # Get weekly returns
        self.weeklyReturns = pricesWed.pct_change().drop(pricesWed.index[0])  # drop first NaN row

    # METHOD COMPUTING ANNUAL RETURNS, ANNUAL STD. DEV. & SHARPE RATIO OF ASSETS
    def __get_stat(self, start, end):

        # ANALYZE THE DATA for a given time period
        weeklyData = self.weeklyReturns[(self.weeklyReturns.index > start) & (self.weeklyReturns.index < end)].copy()

        # Create table with summary statistics
        mu_ga = meanRetAn(weeklyData)                   # Annualised geometric mean of returns
        stdev_a = weeklyData.std(axis=0) * np.sqrt(52)  # Annualised standard deviation of returns
        sharpe = round(mu_ga/stdev_a, 2)                # Sharpe ratio of each financial product

        # Write all results into a data frame
        statDf = pd.concat([mu_ga, stdev_a, sharpe], axis=1)
        statDf.columns = ["Average Annual Returns", "Standard Deviation of Returns", "Sharpe Ratio"]
        statDf["Name"] = statDf.index                   # Add names into the table

        return statDf

    # METHOD TO PLOT THE OVERVIEW OF THE FINANCIAL PRODUCTS IN TERMS OF RISK AND RETURNS
    def plot_dots(self, start, end, ML=None, MLsubset=None):
        # Get statistics for a given time period
        data = self.__get_stat(start, end)

        # IF WE WANT TO HIGHLIGHT THE SUBSET OF ASSETS BASED ON ML
        if ML == "MST":
            setColor = "Type"
            data.loc[:, "Type"] = "The rest of assets"
            for fund in MLsubset:
                data.loc[fund, "Type"] = "Subset based on MST"
        if ML == "Clustering":
            setColor = "Type"
            data.loc[:, "Type"] = MLsubset.loc[:, "Cluster"]
        if ML == None:
            setColor = None

        # PLOTTING Data
        fig = px.scatter(data,
                         x="Standard Deviation of Returns",
                         y="Average Annual Returns",
                         hover_data=["Sharpe Ratio", "Name"],
                         color=setColor,
                         title="The Relationship between Annual Returns and Standard Deviation of Returns from "
                               + start + " to " + end)

        # AXIS IN PERCENTAGES
        fig.layout.yaxis.tickformat = ',.1%'
        fig.layout.xaxis.tickformat = ',.1%'

        fig.show()

    # METHOD TO PREPARE DATA FOR ML AND BACKTESTING
    def setup_data(self, start, end, train_test, train_ratio=0.5):
        self.start = start
        self.end = end
        self.train_test = train_test

        # Get data for a given time interval
        data = self.weeklyReturns[(self.weeklyReturns.index > start) & (self.weeklyReturns.index < end)].copy()

        # IF WE DIVIDE DATASET
        if train_test:
            # DIVIDE DATA INTO TRAINING AND TESTING PARTS
            breakPoint = int(np.floor(len(data.index) * train_ratio))

            # DEFINITION OF TRAINING AND TESTING DATASETS
            self.trainDataset = data.iloc[0:breakPoint, :]
            self.testDataset = data.iloc[breakPoint:, :]

            # Get dates
            self.endTrainDate = str(self.trainDataset.index.date[-1])
            self.startTestDate = str(self.testDataset.index.date[0])

            self.dataPlot = self.__get_stat(start, self.endTrainDate)
            self.lenTest = len(self.testDataset.index)
        else:
            self.trainDataset = data
            self.dataPlot = self.__get_stat(start, end)
            self.lenTest = 0


if __name__ == "__main__":

    # TICKERS OF ETFs WE ARE GOING TO WORK WITH
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

    # INITIALIZATION OF THE CLASS
    algo = TradeBot(start="2015-09-23", end="2019-09-22", assets=tickers)

    # PLOT INTERACTIVE GRAPH
    algo.plot_dots(start="2016-01-01", end="2018-01-01")

    # DIVIDE DATASET INTO TRAINING AND TESTING PART?
    algo.setup_data(start="2015-12-23", end="2018-08-22", train_test=True, train_ratio=0.6)
'''



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



### SCENERIO GENERATION

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


### MATHEMATICAL OPTIMIZATION

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

'''