import pandas as pd 

validation_data = pd.read_csv("D:/Personal Documents/MATHAN PERSONAL DOCUMENTS/Technical Training/Data Science Projects/katacoda-scenarios/fraud-detection-evaluate-model/assets/validation_data.csv")

print(validation_data.head())

actual, predicted = validation_data['actual'], validation_data['predicted']

print(actual.head())

print(predicted.head())

print("THE ACTUAL AND PREDICTED VALUES HAVE BEEN READ INTO A DATAFRAME")
