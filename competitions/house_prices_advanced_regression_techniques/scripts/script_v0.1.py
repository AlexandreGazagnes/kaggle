#!/usr/bin/env python3
# -*- coding: utf-8 -*-



############################################
# 	House price competition
############################################



# v1.2 brute force approche with Linear and polynomial regresion 
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

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV, Ridge, \
								ElasticNet, LassoCV, LassoLarsCV, HuberRegressor
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline




##########################################################
# 	DataFrame creation 
##########################################################


train_df = pd.read_csv("train.csv", index_col=0)

y_train = train_df["SalePrice"]
X_train  = train_df.drop(["SalePrice"], axis=1)


X_test = pd.read_csv("test.csv", index_col=0)



##########################################################
# 	Creation of personalized features 
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


	# print("DF TRAIN")
	# na_col = 0
	# for col in X_train.columns : 
	# 	na = X_train[col].isna().any()
	# 	if na : 
	# 		print("att {} has NaN!".format(col))
	# 		na_col +=1
	# print("number of features with one/more NaN :", na_col)


	# print("DF test")
	# na_col = 0
	# for col in X_test.columns : 
	# 	na = X_test[col].isna().sum()
	# 	if na : 
	# 		print("att {} has NaN! : {}".format(col, na))
	# 		na_col +=1
	# print("number of features with one/more NaN :", na_col)



##########################################################
# 	Regression 
##########################################################

polynomial_features = PolynomialFeatures(degree=3, include_bias=True)
linear_regression = LinearRegression()

model = Pipeline([("polynomial_features", polynomial_features),
				 ("linear_regression", linear_regression)])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred = y_pred.round(2)
	# print(len(y_pred))



##########################################################
# 	Saving results
##########################################################


result = pd.DataFrame(dict(SalePrice=y_pred))
result.index += 1461 ; result.index.name = "ID"

result.to_csv("kaggle_submission.csv")

