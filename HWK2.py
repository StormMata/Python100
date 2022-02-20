# ------------------------------------
# -------- HWK 2 - Regression --------
# ------------ Storm Mata ------------
# ------------------------------------

from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sklearn

# ----------------------------------------------------- Import Data -----------------------------------------------------
FullData  = pd.read_csv('houseSmallData.csv')                                       # Pull data from file
TrainData = FullData.iloc[0:100, :]                                                 # Grad first 100 rows and all columns

# print(TrainData)

# SalePrice = TrainData['SalePrice']
# SalePrice.describe()

# plt.scatter(TrainData['GrLivArea'], y=TrainData['SalePrice'])
# plt.scatter(TrainData['GarageArea'], y=TrainData['SalePrice'])

# ----------------------------------------------------- Clean Data ------------------------------------------------------
# nulls = TrainData.isnull().sum().sort_values(ascending=False)[0:20]
# type(nulls)
data = TrainData.select_dtypes(include=[np.number]).interpolate().dropna(axis=1)
# data.shape

# ----------------------------------------- Determine Correlation of Predictors -----------------------------------------
# numeric = data.select_dtypes(include=[np.number])
# numeric.head()

Indices = list(range(5,39))

for i in Indices:
    print("Regression with the top",i,"correlated variables")
    numeric = data.select_dtypes(include=[np.number])
    numeric.head()

    # numeric = numeric.loc[:, (numeric != 0).any(axis=0)]

    # numeric = numeric.drop(['BsmtFullBath'], axis=1)
    # numeric = numeric.drop(['BsmtHalfBath'], axis=1)

    numeric = numeric.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

    # numeric = numeric.loc[:, (numeric != 0).any(axis=0)]

    corr = numeric.corr()
    cols = corr['SalePrice'].sort_values(ascending=False)[:i].index

    # print(cols)

    TrainX = data[cols]

    # X = X.loc[:, (X != 0).any(axis=0)]
    # X = X.loc[(X != 0).any(axis=1),:]
    # X[(X != 0).all(1)]
    # TrainX = TrainX.loc[(TrainX[:] != 0).all(axis=1)]

    # nans = numeric.sum()
    # print(nans)

    # print(X)
    TrainY = TrainX['SalePrice']
    # print(Y)
    TrainY = np.log(TrainY)
    # print(Y)
    TrainX = TrainX.drop(['SalePrice'], axis=1)

    # --------------------------------------------------- Create a Model ----------------------------------------------------
    lr = linear_model.LinearRegression()
    model = lr.fit(TrainX, TrainY)
    TrainPredictions = model.predict(TrainX)
    print(f"   Training R^2 is: {model.score(TrainX,TrainY)}")

    # plt.hist(Y - predictions)

    # plt.scatter(predictions, Y, color='r')

    # TrainData[['SalePrice', 'OverallQual', 'MasVnrArea']]

    # ----------------------------------------------------- Test Model ------------------------------------------------------
    TestData = pd.read_csv('testData.csv')
    TestX = TestData[cols]
    TestX = TestX.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    # TestData = TestData.loc[(TestData != 0).any(axis=1),:]
    # TestX = TestX.loc[(TestX[:] != 0).all(axis=1)]

    # TestData.shape
    # TestData.head()

    # print(TestX)
    # print("training data", X)
    TestY = TestX['SalePrice']
    TestY = np.log(TestY)
    # print(Y)
    TestX = TestX.drop(['SalePrice'], axis=1)
    # print(X)

    TestPredictions = model.predict(TestX)
    print(f"   Testing  R^2 is: {model.score(TestX,TestY)}\n")
    sklearn.metrics.mean_squared_error(TestPredictions, TestY)
    # predictions