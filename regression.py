import random
from collections import defaultdict

import pandas as pd

#plots
import matplotlib
matplotlib.use('TkAgg')
import seaborn
from pylab import rcParams

# plotting settings
rcParams['figure.figsize'] = 10, 8
seaborn.set_style('whitegrid')

# Regression
import numpy
import sklearn
from sklearn.linear_model import LogisticRegression

"""
Test of logistic regression techniques.

Using Data from:

http://data.princeton.edu/wws509/datasets/#housing
http://data.princeton.edu/wws509/datasets/copen.dat
"""

# Parse copenhagen data to pandas
DATA = 'copenhagen_housing_conditions.dat'

with open(DATA) as data_file:
    data = [line.split() for line in data_file]

_columns = data.pop(0) # discard first line
_data = defaultdict(list)

for row in data:
    _ = row.pop(0) # throw away row #

    for n, c in enumerate(_columns):
        _data[c].append(row[n])

raw_housing = pd.DataFrame(_data) # Data imported to pandas table.

# expand n column to n rows
list_of_series = []
for x, y in raw_housing.iterrows():
    for i in range(int(y['n'])):
        list_of_series.append(y)

# shuffle (for giggles)
random.shuffle(list_of_series)

expanded_housing = pd.DataFrame(list_of_series, index=range(len(list_of_series)))
del expanded_housing['n']

# end data ingestion



# use dummies to switch qualitative data to numeric
dummies = pd.get_dummies(expanded_housing)

# clear out unnecessary columns
del dummies['satisfaction_medium']
del dummies['satisfaction_high']

# Correlation data
corr = dummies.corr()

c_special = 'satisfaction_low'
low_sat_corr = corr[[c_special]].sort_values(c_special, ascending=False).drop(c_special)

print low_sat_corr


# # logistic regression stuff - enables prediction of data points
# feature_columns = ['contact_high',
#                    'contact_low',
#                    'housing_apartments',
#                    'housing_atrium',
#                    'housing_terraced',
#                    'housing_tower',
#                    'influence_high',
#                    'influence_low',
#                    'influence_medium']
#
# X = dummies[feature_columns]
# y = dummies['satisfaction_low']
# # sklearn.cross_validation.train_test_split(X, T, )
#
# model = LogisticRegression()
# model.fit(X, y)

# print "predictive accuracy"
# print model.score(X, y)
#
# print "Null error rate"
# print y.mean()




# plot = seaborn.barplot(low_sat_corr, low_sat_corr.index.values)
# plot.figure.savefig("output.png")



