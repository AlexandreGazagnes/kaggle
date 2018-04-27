#!/usr/bin/env python3
# -*- coding: utf-8 -*-



############################################
# 	House price competition
############################################



# v2.0 lassoCV approch with a brute force cleaning and reshaping of features 
# notinhg good just a firt try
# best is yet to come ! 



###############################################
# 	Import 
###############################################


import pandas as pd 
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV, Ridge, Lasso, \
								ElasticNet, LassoCV, LassoLarsCV, HuberRegressor
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV



##########################################################
# 	DataFrame creation 
##########################################################


train_df = pd.read_csv("train.csv", index_col=0)

y_train = train_df["SalePrice"]
X_train  = train_df.drop(["SalePrice"], axis=1)


X_test = pd.read_csv("test.csv", index_col=0)



##########################################################
# 	Data cleaning : missing values  - NaN and Null
##########################################################


for X in (X_train, X_test ) : 

	X["p_TotSurf"] = X["TotalBsmtSF"] + X["GrLivArea"]
	X["p_TotSurfWithPool"] = X["p_TotSurf"] + X["PoolArea"]
	X["p_TotalBath"] = X["FullBath"]+ X["HalfBath"]+ \
				X["BsmtFullBath"]+ X["BsmtHalfBath"]

	X.drop(	["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"], 
						axis=1, inplace=True)
	X.drop(	['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'], 
						axis=1, inplace=True)


my_features = ['p_TotSurf', 'GrLivArea', 'YrSold', 'GarageArea', 'OverallQual', 
					'YearRemodAdd', 'p_TotalBath', 'YearBuilt']

X_train = X_train.loc[:, my_features]
X_test = X_test.loc[:, my_features]


	# print(X_train.shape)
	# print(X_test.shape)
	# print(X_train.head())
	# print(X_test.head())



##########################################################
# 	Data cleaning 
##########################################################


# data cleaning : drop na from X_test
# filling with mean 
for col in X_test.columns : 
	X_test[col] = X_test[col].fillna(X_test[col].median())



# scaler data

std_scaler = StandardScaler().fit(X_train) 
X_train = std_scaler.transform(X_train)

std_scaler = StandardScaler().fit(X_test) 
X_test = std_scaler.transform(X_test)



##########################################################
# model selection
#########################################################


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25)


def rmsle(y_pred, y_test) : 
	assert len(y_test) == len(y_pred)
	assert (y_pred < 0).sum() == 0
	return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))


# try linear regression

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
baseline_error = rmsle(y_test, y_pred)
print(baseline_error)


n_alphas = 100
alphas = np.logspace(-5, 5, 200 )

coefs = list()
errors = list()


for a in alphas : 
	ridge = Ridge(alpha=a)
	clf = GridSearchCV(ridge,{}, cv=10)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	errors.append(round(rmsle(y_test, y_pred),4))  





plt.plot(alphas, errors)
plt.plot(alphas, [baseline_error for _ in alphas])
plt.xscale('log')
plt.ylim([0.1, 0.3])
plt.show()


errors = list(zip(alphas, errors))
errors.sort(key=lambda x : x[1])
print(errors[:10])	


input()



		# ##########################################################
		# # 	Regression 
		# ##########################################################


		# for a in range(900, 1000, 10) : 
		# 	ridge = Ridge(alpha=a)
		# 	clf = GridSearchCV(ridge,{}, cv=10)
		# 	clf.fit(X_train, y_train)
		# 	y_pred = clf.predict(X_test)
		# 	y_pred = y_pred.round(2)



		# 	##########################################################
		# 	# 	Saving results
		# 	##########################################################


		# 	result = pd.DataFrame(dict(SalePrice=y_pred))
		# 	result.index += 1461 ; result.index.name = "ID"

		# 	result.to_csv("kaggle_submission_{}.csv".format(a))

