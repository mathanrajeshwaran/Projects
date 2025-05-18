import pandas as pd
from collections import Counter
from scipy import stats
import numpy as np 

df = pd.read_csv("D:/Personal Documents/MATHAN PERSONAL DOCUMENTS/Technical Training/Data Science Projects/katacoda-scenarios/fraud-detection-data-prep/assets/fraud_detection_data.csv")

df['card_number'] = df['card_number'].astype(str)

print(df.head())

print(df.info())

print(df.describe())

print(df.columns)

print(Counter(df['fraud_flag']))

import ast 

df['merchant_state'] = df['merchant_state'].astype('category')
df['merchant_state_code'] = df['merchant_state'].cat.codes

df['merchant_city'] = df['merchant_city'].astype('category')
df['merchant_city_code'] = df['merchant_city'].cat.codes


df['card_type'] = df['card_type'].astype('category')
df['card_type_code'] = df['card_type'].cat.codes


df['cardholder_name'] = df['cardholder_name'].astype('category')
df['cardholder_name_code'] = df['cardholder_name'].cat.codes

number_of_items = [len(ast.literal_eval(x)) for x in list(df['items'])]

df['number_of_items'] = number_of_items

threshold = 3
z_scores = np.abs(stats.zscore(df['transaction_amount']))
df_no_outliers = df[(z_scores < threshold)]

print("CATEGORICAL VARIABLES HAVED BEEN ENCODED AND OUTLIERS HAVE BEEN REMOVED")

features  = ['merchant_state_code','merchant_city_code', 'card_type_code','cardholder_name_code',
             'transaction_amount', 'number_of_items']
target = 'fraud_flag'

X = df_no_outliers[features]

y = df_no_outliers[target]

X.to_csv("features.csv", index=False)

y.to_csv("targets.csv", index=False)

print("FEATURES AND TARGETS HAVE BEEN WRITTEN TO CSV FILES")
