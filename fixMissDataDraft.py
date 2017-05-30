# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
#fill in NA values 
filepath = "C:\\Users\\Abhijit\\Documents\\Abhijit Work\\Draft Value\\Pick Draft Data.csv"
def handle_data(filepath):
    data = pd.read_csv(filepath)
    tovdata = data.loc[data["tov_per_g"] != 0]
    value = np.mean(tovdata["tov_per_g"])
    data["tov_per_g"] = data["tov_per_g"].apply(lambda x: value if x == 0 else x)
    return data

def buildModel(data):
    
    lasso = Lasso(fit_intercept = False)
    inputs = data.loc[data["lb"]!=0]["height"]
    inputs = np.reshape(inputs,(inputs.shape[0],1))
    output = data.loc[data["lb"]!=0]["lb"]
    output = np.reshape(output,(output.shape[0],1))
    lasso.fit(inputs,output)
    return lasso
def fillMissWeights(data,lasso):
    for item in range(0,data.shape[0]):
        if data.iloc[item, data.columns.get_loc("lb")] == 0:
           data.iloc[item, data.columns.get_loc("lb")] = lasso.predict(data.iloc[item,data.columns.get_loc("height")])[0][0] 
    return data

data_1 = handle_data(filepath)
lasso = buildModel(data_1)
data_2 = fillMissWeights(data_1,lasso)

def writeToCSV(data_2):
    data_2.to_csv("Pick Draft Data (Filled 2).csv",header = data_2.columns.values)
    return None

writeToCSV(data_2)
