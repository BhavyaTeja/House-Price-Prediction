#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 16:32:57 2017

@author: bhavyateja
"""

#Importing the packages

import pandas as pd
import sklearn as sk
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

#Importing the test and train data


Train_data = pd.read_csv('/Users/bhavyateja/Github_Projects/House-Price-Prediction/Dataset/train.csv')
Test_data = pd.read_csv('/Users/bhavyateja/Github_Projects/House-Price-Prediction/Dataset/test.csv')

#Listing the columns with character data types in both test and training datasets

Character_columns_Train = [col for col in Train_data.columns if is_string_dtype(Train_data[col]) == True]
Character_columns_Test = [col for col in Train_data.columns if is_numeric_dtype(Train_data[col]) == True]