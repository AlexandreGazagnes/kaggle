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

test_df = pd.read_csv("test.csv", index_col=0)



##########################################################
# 	Data cleaning : missing values  - NaN and Null
##########################################################


# delete rows with all NaN values

train_df.dropna(axis=1, how="all", inplace=True)

test_df.dropna(axis=1, how="all", inplace=True)

# drop if any NaN

train_df.dropna(axis=1, how='any', inplace=True)

test_df.drop(['TotRmsAbvGrd'], axis=1, inplace=True)
test_df.drop(["LotFrontage"], axis=1, inplace=True)



# deleting missing data
try : 
	train_df = train_df.drop((missing_data[missing_data['Total'] > 1]).index,1)
	train_df = train_df.drop(train_df.loc[train_df['Electrical'].isnull()].index)
except : 
	print("value already deleted	")

try : 
	test_df = test_df.drop((missing_data[missing_data['Total'] > 1]).index,1)
	test_df = test_df.drop(test_df.loc[test_df['Electrical'].isnull()].index)
except : 
	print("value already deleted")

try : 

	#deleting outliers
	train_df = train_df[train_df["GrLivArea"]  < 4550]
	train_df = train_df[train_df["TotalBsmtSF"]  < 3000]
	train_df = train_df.drop(["GarageCars"], axis=1)
	train_df = train_df[train_df["GarageArea"]  < 1200]
	train_df = train_df.drop("TotRmsAbvGrd", axis=1)


except : 
	print("value already deleted")


##########################################################
# 	Preprocessing : creating personnal features
##########################################################


# first remove personal useless features 

useless_features = ['LotConfig', 'Alley', 'LotShape', 'OverallCond', 'Condition2', 
	'LowQualFinSF', 'Electrical','OpenPorchSF','EnclosedPorch','3SsnPorch',
	'ScreenPorch','RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 
	'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 
	'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
	'BsmtFinType2','BsmtFinSF2','BsmtUnfSF', "FireplaceQu", "GarageType",
	"GarageYrBlt", "GarageFinish", "GarageCars", "GarageQual", "GarageCond",
	"BsmtFullBath","BsmtHalfBath" ]

for feature in useless_features : 
	try : 
		train_df = train_df.drop([feature], axis=1)
	except : 
		print("{} already deleted :) ".format(feature))


for feature in useless_features : 
	try : 
		test_df = test_df.drop([feature], axis=1)
	except : 
		print("{} already deleted :) ".format(feature))



# droping non numerical columns
	numeric_cols = [col for col in train_df.select_dtypes(include=[np.number]).columns.tolist()]
	cat_cols = [col for col in train_df.select_dtypes(exclude=[np.number]).columns.tolist()]

	train_df.drop(cat_cols, axis=1, inplace=True)

	numeric_cols = [col for col in test_df.select_dtypes(include=[np.number]).columns.tolist()]
	cat_cols = [col for col in test_df.select_dtypes(exclude=[np.number]).columns.tolist()]

	test_df.drop(cat_cols, axis=1, inplace=True)


# data cleaning : drop na from test_df
# filling with mean 
for col in test_df.columns : 
	test_df[col] = test_df[col].fillna(test_df[col].median())


# splitin train/test

y_train = train_df["SalePrice"]
X_train = train_df.drop(["SalePrice"], axis=1)
X_test = test_df


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

errors=list()

ridge = Ridge()
params = {"alpha":alphas}
gridge = GridSearchCV(ridge, params, cv=10)
gridge.fit(X_train, y_train)
y_pred = gridge.predict(X_test)
errors.append(round(rmsle(y_test, y_pred),4))  

print(errors)



# plt.plot(alphas, errors)
# plt.plot(alphas, [baseline_error for _ in alphas])
# plt.xscale('log')
# plt.ylim([0.1, 0.3])
# plt.show()


# errors = list(zip(alphas, errors))
# errors.sort(key=lambda x : x[1])
# print(errors[:10])	


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

