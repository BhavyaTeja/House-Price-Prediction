#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 16:32:57 2017

@author: bhavyateja
"""

#Importing the packages

import seaborn as sns
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

#Importing the test and train data


Train_data = pd.read_csv('/Users/bhavyateja/Github_Projects/House-Price-Prediction/Dataset/train.csv')
Test_data = pd.read_csv('/Users/bhavyateja/Github_Projects/House-Price-Prediction/Dataset/test.csv')

#Summary of the datasets

Train_data.describe()

#Listing the columns with character data types in both test and training datasets

Character_columns_Train = [col for col in Train_data.columns if is_string_dtype(Train_data[col]) == True]
Character_columns_Test = [col for col in Train_data.columns if is_numeric_dtype(Train_data[col]) == True]
Numeric_columns_Train = [col for col in Train_data.columns if is_numeric_dtype(Train_data[col]) == True]

#Adding the column Sale price to the Test dataset

Test_data['SalePrice'] = pd.Series(np.NaN, index = Test_data.index)

#Analyzing the Sale Price column in Train Data and visualizing it 

Train_data['SalePrice'].describe()

plt.hist(Train_data['SalePrice'])
plt.xlabel('Sale price of the house')
plt.show()

data = pd.concat([Train_data['SalePrice'], Train_data['GrLivArea']], axis=1)
data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000));

#Seperating the character and numeric columns for data visualization

Train_data_char = Train_data[Character_columns_Train]
Train_data_num = Train_data[Numeric_columns_Train]

#correlation matrix

corrmat = Train_data.corr()
f, ax = plt.subplots(figsize=(15, 12))
sns.heatmap(corrmat, vmax=.8, square=True);

#Preprocessing the data by combining both Test and Train data

Train_data['isTrain'] = pd.Series(1, index = Train_data.index)
Test_data['isTrain'] = pd.Series(0, index = Test_data.index)

house = pd.concat([Train_data, Test_data])

house['MasVnrArea'] = house.MasVnrArea.replace(np.NaN, np.mean(house['MasVnrArea']))            #Replacing null values with mean values

house['LotFrontage'] = house.LotFrontage.replace(np.NaN, np.nanmedian(house['LotFrontage']))    #Replacing null values with median values

house['Street'] = house['Street'].astype('category')

#Label Encoder implement here*****
house[Character_columns_Train] = house[Character_columns_Train].astype('category')
house[Character_columns_Train] = house[Character_columns_Train].apply(lambda x: x.cat.codes)
