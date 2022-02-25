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
FullData  = pd.read_csv('train.csv')                                                # Import training data
TrainData = FullData.iloc[0:1000, :]                                                # Grab first 1000 rows and all columns

# ----------------------------------------------------- Clean Data ------------------------------------------------------

data = TrainData.select_dtypes(include=[np.number]).interpolate().dropna(axis=1)    # Only include numbers

# ----------------------------------------- Determine Correlation of Predictors -----------------------------------------

numeric = data.select_dtypes(include=[np.number])
numeric.head()

numeric = numeric.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

corr = numeric.corr()                                                               # Get correlations of variables
cols = corr['SalePrice'].sort_values(ascending=False)[:10].index                    # Sort top 10 correlated variables

TrainX = data[cols]                                                                 # Grab 10 variables

TrainY = TrainX['SalePrice']                                                        # Set dependent variable

TrainX = TrainX.drop(['SalePrice'], axis=1)                                         # Remove dependent variable

# --------------------------------------------------- Create a Model ----------------------------------------------------

lr = linear_model.LinearRegression()                                                # Set regression model
model = lr.fit(TrainX, TrainY)                                                      # Define model from training data
TrainPredictions = model.predict(TrainX)                                            # Predict dependent variables

print(f"   Training R^2 is: {model.score(TrainX,TrainY)}")

# ----------------------------------------------------- Test Model ------------------------------------------------------

TestData = FullData.iloc[1001:1459, :]                                              # Grab testing data; last 458 rows

TestX = TestData[cols]                                                              # Get same top correlated variables

TestY = TestX['SalePrice']                                                          # Set dependent variable

TestX = TestX.drop(['SalePrice'], axis=1)                                           # Remove dependent variable

TestPredictions = model.predict(TestX)                                              # Use model to predict dependent variables
print(f"   Testing  R^2 is: {model.score(TestX,TestY)}\n")
sklearn.metrics.mean_squared_error(TestPredictions, TestY)

IDs = TestData['Id']

file = open("predictions.csv", "w")                                                 # Open csv file
writer = csv.writer(file)

writer.writerow(['Id', 'SalePrice'])                                                # Write headers

for w in range(1001,1458):
    writer.writerow([IDs[w], TestPredictions[w-1001]])                              # Write IDs and predictions

file.close()