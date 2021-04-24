#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 10:17:02 2020

@author: Petr Vanek
"""

import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = "browser"



"""
    ----------------------------------------------------------------------
    Data Analytics and Visualisation: PLOTTING INTERACTIVE GRAPH
    ----------------------------------------------------------------------
"""    
def plotInteractive(data, ML, MLsubset, start, end):
    # IF WE WANT TO HIGHLIGHT THE SUBSET OF ASSETS BASe ON ML
    if ML == "MST":
        setColor= "Type"
        data.loc[:,"Type"] = "The rest of assets"
        for fund in MLsubset:
             data.loc[fund,"Type"] = "Subset based on MST"
    if ML == "Clustering":
        setColor = "Type"
        data.loc[:,"Type"] = MLsubset.loc[:,"Cluster"]
    if ML == None:
        setColor = None
    
    # ROUND THE SHARPE RATIO
    data["Sharpe Ratio"] = round(data["Sharpe Ratio"],2)

    # PLOTTING Data
    fig = px.scatter(data, 
                     x="Standard Deviation of Returns",
                     y="Average Annual Returns", 
                     hover_data=["Sharpe Ratio", "Name"],
                     color= setColor,
                     title="The Relationship between Annual Returns and Standard Deviation of Returns from "
                            + start + " to " + end)

    
    # AXIS IN PERCENTAGES
    fig.layout.yaxis.tickformat = ',.1%'
    fig.layout.xaxis.tickformat = ',.1%'


    fig.show()
    
    


"""
    ----------------------------------------------------------------------
    Mathematical Optimization: PLOTTING PERFORMANCE AND COMPOSITION GRAPH
    ----------------------------------------------------------------------
"""       

def plotOptimization(performance, performanceBenchmark, composition, names):
    # PERFORMANCE
    performance.index = performance.index.date
    df_to_plot = pd.concat([performance, performanceBenchmark], axis =1)
    fig = px.line(df_to_plot, x=df_to_plot.index, y=df_to_plot.columns,
              title='Comparison of different strategies')
    fig.show()

    
    # COMPOSITION
    composition.columns = list(names)
    composition = composition.loc[:, (composition != 0).any(axis=0)]
    data = []
    for isin in composition.columns:
        trace = go.Bar(
                x = composition.index,
                y = composition[isin],
                name = str(isin)
                )
        data.append(trace)
    
    layout = go.Layout(barmode = 'stack')
    fig = go.Figure(data = data, layout = layout)
    fig.update_layout(
        title="Portfolio Composition",
        xaxis_title="Number of the Investment Period",
        yaxis_title="Composition",
        legend_title="Name of the Fund")
    fig.layout.yaxis.tickformat = ',.1%'
    fig.show()
    

    
    
    
    