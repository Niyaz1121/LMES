from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("petrol_consumption.csv")
print(df.head())

df_columns = df.columns
"""commented by mohan"""
##print(df_columns)

x = df[["Petrol_tax", "Average_income", "Paved_Highways","Population_Driver_licence(%)"]]
y = df["Petrol_Consumption"]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

# model = DecisionTreeRegressor(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42)
# model.fit(x_train, y_train)
# y_prediction = model.predict(x_test)
# print(y_test)
#
# accuracy_score = r2_score(y_test, y_prediction)
# print(accuracy_score)


model = DecisionTreeRegressor(criterion='squared_error',max_depth=5, random_state=42)
model.fit(x_train, y_train)
y_prediction = model.predict(x_test)
print(y_test)

accuracy_score = r2_score(y_test, y_prediction)
print(accuracy_score)

