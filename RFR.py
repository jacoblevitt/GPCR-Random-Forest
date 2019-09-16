import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

dataX_train = np.array(pd.read_csv('dataX_train.csv'))
dataY_train = np.array(pd.read_csv('dataY_train.csv'))

dataX_test = np.array(pd.read_csv('dataX_test.csv'))
dataY_test = np.array(pd.read_csv('dataY_test.csv'))

rf = RandomForestRegressor(n_estimators = 550, max_depth = 20, random_state = 2, verbose = 1)
rf.fit(dataX_train, dataY_train)

preds = rf.predict(dataX_test)
acc = r2_score(dataY_test, preds, multioutput = 'variance_weighted')
print(acc)

