# ------------------------------------
# -------- HWK 2 - Regression --------
# ------------ Storm Mata ------------
# ------------------------------------

from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sklearn
import csv

# ----------------------------------------------------- Import Data -----------------------------------------------------
FullData  = pd.read_csv('train.csv')
TrainData = FullData.iloc[0:1000, :]                                                 # Grad first 100 rows and all columns

# ----------------------------------------------------- Clean Data ------------------------------------------------------

data = TrainData.select_dtypes(include=[np.number]).interpolate().dropna(axis=1)

# ----------------------------------------- Determine Correlation of Predictors -----------------------------------------

Indices = list(range(5,39))

# for i in Indices:
# print("Regression with the top",i,"correlated variables")
numeric = data.select_dtypes(include=[np.number])
numeric.head()

numeric = numeric.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

corr = numeric.corr()
cols = corr['SalePrice'].sort_values(ascending=False)[:10].index

TrainX = data[cols]

TrainY = TrainX['SalePrice']

TrainX = TrainX.drop(['SalePrice'], axis=1)

# --------------------------------------------------- Create a Model ----------------------------------------------------
lr = linear_model.LinearRegression()
model = lr.fit(TrainX, TrainY)
TrainPredictions = model.predict(TrainX)

print(f"   Training R^2 is: {model.score(TrainX,TrainY)}")

# ----------------------------------------------------- Test Model ------------------------------------------------------

TestData = FullData.iloc[1001:1459, :]  

TestX = TestData[cols]

TestY = TestX['SalePrice']

TestX = TestX.drop(['SalePrice'], axis=1)

TestPredictions = model.predict(TestX)
print(f"   Testing  R^2 is: {model.score(TestX,TestY)}\n")
sklearn.metrics.mean_squared_error(TestPredictions, TestY)

IDs = TestData['Id']
print(IDs)
print(TestPredictions)

file = open("predictions.csv", "w")
writer = csv.writer(file)

writer.writerow(['Id', 'SalePrice'])

for w in range(1001,1458):
    writer.writerow([IDs[w], TestPredictions[w-1001]])

file.close()