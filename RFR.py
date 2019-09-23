import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import *

dataX_train = np.loadtxt('dataX_train.csv')
dataY_train = np.loadtxt('dataY_train.csv')
dataX_test = np.loadtxt('dataX_test.csv')
dataY_test = np.loadtxt('dataY_test.csv')

# linear regression

reg = LinearRegression()
reg.fit(dataX_train, dataY_train)
preds = reg.predict(dataX_test)

# evals

evs = explained_variance_score(dataY_test, preds)
print()
print('LR explained_variance_score = %s' %evs)

mae = mean_absolute_error(dataY_test, preds)
print()
print('LR mean_absolute_error = %s' %mae)

mse = mean_squared_error(dataY_test, preds)
print()
print('LR mean_squared_error = %s' %mse)

acc = r2_score(dataY_test, preds)
print()
print('LR r2_score = %s' %acc)

# first pass Random Forest

rf = RandomForestRegressor(n_estimators = 100, max_depth = None, max_features = 'log2', random_state = 2, n_jobs = -1, verbose = 1)
rf.fit(dataX_train, dataY_train)

preds = rf.predict(dataX_test)

# evals

evs = explained_variance_score(dataY_test, preds)
print()
print('RF explained_variance_score = %s' %evs)

mae = mean_absolute_error(dataY_test, preds)
print()
print('RF mean_absolute_error = %s' %mae)

mse = mean_squared_error(dataY_test, preds)
print()
print('RF mean_squared_error = %s' %mse)

acc = r2_score(dataY_test, preds)
print()
print('RF r2_score = %s' %acc)

# feature importances

alphabet = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}
keys = list(alphabet.keys())

importances = rf.feature_importances_.tolist()
features = (-rf.feature_importances_).argsort().tolist()

for i in features[:10]:
    residue = keys[(i%20) - 1] if i%20 != 0 else 0
    position = i//20
    importance = str(importances[i])
    print('%s%d -- %s' % (residue, position, importance))

# second pass

features_new = [i for i in features if importances[i] != 0]

dataX_train = np.delete(dataX_train, features_new, axis = 1)
dataX_test = np.delete(dataX_test, features_new, axis = 1)

rf = RandomForestRegressor(n_estimators = 100, max_depth = None, max_features = 'log2', random_state = 2, n_jobs = -1, verbose = 1)
rf.fit(dataX_train, dataY_train)

preds = rf.predict(dataX_test)

# evals

evs = explained_variance_score(dataY_test, preds)
print()
print('RF explained_variance_score = %s' %evs)

mae = mean_absolute_error(dataY_test, preds)
print()
print('RF mean_absolute_error = %s' %mae)

mse = mean_squared_error(dataY_test, preds)
print()
print('RF mean_squared_error = %s' %mse)

acc = r2_score(dataY_test, preds)
print()
print('RF r2_score = %s' %acc)
    
    



