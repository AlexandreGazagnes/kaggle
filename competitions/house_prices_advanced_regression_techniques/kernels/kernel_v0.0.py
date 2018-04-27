#!/usr/bin/env python3
# -*- coding: utf-8 -*-



############################################
# 	House price competition
############################################



# Find here my work regarding my dirst kaggle competition
# This is a kernel, not a script, 
# found here all perparatory work, studies and model selection
# before running a script and submitings predictions

# v0.0 brute force apporche, not factorised, without extern help
# without MOOC implementation, just first "intention"



###############################################
# 	Import 
###############################################


import pandas as pd 
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, Ridge, HuberRegressor
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


print("""
############################################################
# 	1st Part :   DataFrame creation and first prints
############################################################
""")


# df creation

df = pd.read_csv("train.csv", index_col=0)


	# first prints

	# print(df.head())
	# print(df.tail())


	# df global information

	# print(df.describe())
	# print(df.dtypes)
print(df.shape)
	# print(df.ndim)



print("""
##########################################################
# 	2nd Part :   First data exploration
##########################################################
""")


	# # try to print various features 

	# df.hist(bins=50, figsize=(12,8))



print("""
##########################################################
# 	3rd Part :   Data cleaning
##########################################################
""")


# delete rows with all NaN values

df.dropna(axis=1, how="all", inplace=True)
df.dropna(axis=1, how="all", inplace=True)


# any NaN?

na_col = 0
for col in df.columns : 
	na = df[col].isna().any()
	if na : 
		print("att {} has NaN!".format(col))
		na_col +=1
print("number of features with one/more NaN :", na_col)


# drop Na

df.dropna(axis=1, how='any', inplace=True)
print(df.shape)


# any Null?

null_col = 0
for col in df.columns : 
	null = df[col].isnull().all()
	if null : 
		print("att {} has only Null values ? : {}".format(col, null))
		null_col +=1
print("number of features with only null :", null_col)


# detect and delete outliers 

		# for col in df.select_dtypes(include=[np.number]).columns.tolist() : 
		# 	print("{} is numeric : {}".format(col, df[col].dtype))
		# 	col_mean, col_std = df[col].mean(), df[col].std()
		# 	df[col] = df[col].where((np.abs(df[col] - col_mean) \
		# 				/ col_std) > 1, np.nan)


		# desc = df[col].describe()
		# IQ = desc["75%"] -  desc["25%"]
		# print(q1,q3) 



print("""
##########################################################
# 	4rd Part :   Data Preprocessing
##########################################################
""")


# just keep numeric columns

numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns.tolist()]
cat_cols = [col for col in df.select_dtypes(exclude=[np.number]).columns.tolist()]


print(numeric_cols)
print(len(numeric_cols))
print(cat_cols)
print(len(cat_cols))


# create a numeric dataframe

numeric_df = df.drop(cat_cols, axis=1)





numeric_df["p_TotSurf"] = numeric_df["TotalBsmtSF"] + numeric_df["GrLivArea"]
numeric_df["p_TotSurfWithPool"] = numeric_df["p_TotSurf"] + numeric_df["PoolArea"]

numeric_df["p_TotalBath"] = numeric_df["FullBath"]+ numeric_df["HalfBath"]+ \
			numeric_df["BsmtFullBath"]+ numeric_df["BsmtHalfBath"]


numeric_df.drop(	["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"], 
					axis=1, inplace=True)

numeric_df.drop(	['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'], 
					axis=1, inplace=True)

print(numeric_df.columns)
print(numeric_df.shape)

input("continue?")


# create X and y

X = numeric_df.drop(["SalePrice"], axis=1)
y = df["SalePrice"]


# create train/test datasets

X_train, X_test, y_train, y_test = train_test_split(X,y)

# be sure of the dataset's shapes

for k in (X_train, X_test, y_train, y_test) : 
	print(k.shape)



print("""
##########################################################
# 	5th Part :   Brutal Ridge CV Regression
##########################################################
""")


# first Try a brutal approch with ridge regression : 

def one_ridge_CV(norm, CV, *train_test_data) : 
	X_train, X_test, y_train, y_test  = train_test_data
	model = RidgeCV(normalize=norm, cv=CV)
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	score = round(mean_squared_error(y_test, y_pred),4)**0.5
	return score


	# score, y_pred = one_ridge_CV(False, 10, *train_test_split(X,y))
	# print("Brutal score = {}".format(score))


def plot_results(y_pred, y_test, score, main_title): 

	results = pd.DataFrame(dict(val=y_test, pred=y_pred))
	results.sort_values(by=["pred"], inplace=True)
	results.index = range(len(results))

	plt.scatter(results.index, results.val, c="red", label="val", marker=".")
	plt.plot(results.index, results.pred, c="green", label="pred")
	plt.legend() 
	plt.title("{} with score : {}".format(main_title, score))
	plt.show()

# plot_results(y_pred, y_test, score, "Brutal RidgeCV")


# just for fun : 

def multiple_ridge_CV(numb, norm, CV, X,y) : 
	scores = list()
	for  i in range(numb) : 
			score = one_ridge_CV(norm, CV, *train_test_split(X,y))
			scores.append(score*100)

	scores = pd.Series(scores)
	# scores.hist() ; plt.show()
	# print("\nRidge_CV * {} for norm={}, CV={}".format(numb, norm, CV))
	# print(scores.describe())
	# print("\n")
	return scores.describe()


# let's evaluate 100 ridge_cv, 
def evaluate_multi_ridge_cv(J, I, norm, CV, X, y) : 
	scores_list = list()
	for j in range(J) : 
		# print("round {}".format(j)) # , end=" / ")
		scores_df = multiple_ridge_CV(I, True, 10, X, y)
		scores_list.append((j, scores_df))

	scores_df = pd.DataFrame(	[j for i,j in scores_list], 
								index = [i for i,j in scores_list])

	print("\nRidge for {}*{} round for norm {}, CV {} :  "\
			.format(J, I, norm, CV) )
	sub_scores_df = scores_df.loc[:, ["mean", "std", "50%"]]
	sub_scores_df.columns = [i.upper() for i in sub_scores_df.columns]
	desc = sub_scores_df.describe()

	dummy_score =  (	desc.loc["mean", "MEAN"] +desc.loc["mean", "50%"]\
					  +	desc.loc["50%", "MEAN"]+ desc.loc["50%","50%"]) /4 
	dummy_score = round(dummy_score, 4)
	print("score : {}\n".format(dummy_score))
	return dummy_score



# evaluate_multi_ridge_cv(3,10, True, 10, X, y)
# evaluate_multi_ridge_cv(3,10, False, 10, X, y)


input("Continuer?")


print("""
##########################################################
# 	6th Part :   Find Few best features
##########################################################
""")


# lets have a ridge regression but with less feature
# we will try to find the best features 

def evaluate_best_features(X,y) : 
	
	X_train, X_test, y_train, y_test = train_test_split(X,y)

	best_col_list = list()
	for col in X.columns : 

		X_train_col, X_train_test = X_train[col], X_test[col] 

		model = LinearRegression()
		model.fit(X_train_col[:, np.newaxis], y_train)
		col_score = model.score(X_train_test[:, np.newaxis], y_test)
		best_col_list.append([round(col_score,4), col])

	best_col_list.sort(reverse=True)
	return best_col_list

best_col_list = evaluate_best_features(X,y)
print(best_col_list)
print()


# Ok easy for 1 round but for  100 ? 
# lets organize a "tournament"

# intiate a empty dict (idem collections.default_dict)
best_col_dict = dict()
for col in  X.columns : 
	best_col_dict[col] = 0


# initiate points retribution :model 1 

points = [50, 30, 20, 15, 10, 8, 6, 4, 2]

# Loop
for _ in range(100) : 
	best_col_list = evaluate_best_features(X,y)

	for i in range(8) : 
		feature = best_col_list[i][1]
		best_col_dict[feature] += points[i]


best_col_list = [(j, i) for i, j in best_col_dict.items()]
best_col_list.sort(reverse=True)
print(best_col_list)
print()


# re-empty dict 

best_col_dict = dict()
for col in  X.columns : 
	best_col_dict[col] = 0


# initiate points retribution :model 2 

points = [18, 16, 14, 12, 10, 8, 6, 4, 2]

# Loop
for _ in range(100) : 
	best_col_list = evaluate_best_features(X,y)

	for i in range(8) : 
		feature = best_col_list[i][1]
		best_col_dict[feature] += points[i]


best_col_list = [(j, i) for i, j in best_col_dict.items()]
best_col_list.sort(reverse=True)
print(best_col_list)
print()


# OK thi is our  10 best features 

best_col_list = [j for i, j in best_col_list[:10]]
print(best_col_list)
print()



print("""
##########################################################
# 	7th Part :   try new regression with 10 features
##########################################################
""")


# #  first Try : 
# auto_best_col_list = best_col_list
# X1 = X.loc[:, auto_best_col_list]
# evaluate_multi_ridge_cv(10, 10, False, 10, X1, y)


score_list = list()

# for i in range(1, len(auto_best_col_list)) : 
# 		sub_best_col_list = auto_best_col_list[:i]
# 		print("{} -> {}".format(i, sub_best_col_list))
# 		sub_X = X.loc[:, sub_best_col_list]
# 		score = evaluate_multi_ridge_cv(10, 10, False, 10, sub_X, y)
# 		score_list.append((score,"RidgeCV 1D", len(sub_best_col_list), sub_best_col_list))




print("""
##########################################################
# 	8th Part :   try new regression with MY features
##########################################################
""")


#  arbitrary try : 

tot_features = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
			'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
			'GrLivArea', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
			'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
			'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
			'MoSold', 'YrSold', 'SalePrice', 'p_TotSurf', 'p_TotSurfWithPool',
			'p_TotalBath'],


best_features = ['p_TotSurf', 'OverallQual', 'p_TotSurfWithPool', 
				'GrLivArea', 'GarageCars', 'GarageArea', 'p_TotalBath', 
				'TotalBsmtSF', '1stFlrSF', 'TotRmsAbvGrd']

my_features = ["OverallQual", 'GarageArea', 'p_TotalBath', 'p_TotSurf', 
				'GrLivArea', "PoolArea", 'YearRemodAdd', "YearBuilt", 'YrSold']



# for _ in range(20) : 

# 	for i in range(1, len(my_features)) : 
# 			sub_best_col_list = my_features[:i]
# 			print("{} -> {}".format(i, sub_best_col_list))
# 			sub_X = X.loc[:, sub_best_col_list]
# 			score = evaluate_multi_ridge_cv(3, 10, False, 10, sub_X, y)
# 			score_list.append((score, len(sub_best_col_list), sub_best_col_list))


# 	for j in range(0, len(my_features)-1) : 
# 			sub_best_col_list = my_features[j:]
# 			print("{} -> {}".format(i, sub_best_col_list))
# 			sub_X = X.loc[:, sub_best_col_list]
# 			score = evaluate_multi_ridge_cv(3, 10, False, 10, sub_X, y)
# 			score_list.append((score, len(sub_best_col_list), sub_best_col_list))

# 	for i in range(0, len(my_features)):
# 		try : 
# 			j = len(my_features) - i
# 			sub_best_col_list = my_features[i:j]
# 			print("{} -> {}".format(i, sub_best_col_list))
# 			sub_X = X.loc[:, sub_best_col_list]
# 			score = evaluate_multi_ridge_cv(3, 10, False, 10, sub_X, y)
# 			score_list.append((score, "RidgeCV 1D",len(sub_best_col_list), sub_best_col_list))
# 		except : 
# 			break


# 	my_features = shuffle(my_features)


# score_list.sort()

# print(score_list[:20])


results = [
 (3687551.5804, 8, ['p_TotSurf', 'GrLivArea', 'YrSold', 'GarageArea', 'OverallQual', 'YearRemodAdd', 'p_TotalBath', 'YearBuilt']),
 (3696526.2922, 9, ['GrLivArea', 'PoolArea', 'YrSold', 'OverallQual', 'p_TotSurf', 'p_TotalBath', 'GarageArea', 'YearBuilt', 'YearRemodAdd']),
 (3700434.3873, 9, ['OverallQual', 'GarageArea', 'p_TotalBath', 'p_TotSurf', 'YearBuilt', 'YearRemodAdd', 'GrLivArea', 'YrSold', 'PoolArea']),
 (3702826.5642, 8, ['OverallQual', 'GarageArea', 'p_TotalBath', 'p_TotSurf', 'YearBuilt', 'YearRemodAdd', 'GrLivArea', 'YrSold']),
 (3713513.1029, 8, ['YearRemodAdd', 'YearBuilt', 'OverallQual', 'GrLivArea', 'GarageArea', 'p_TotSurf', 'PoolArea', 'p_TotalBath']),
 (3738539.4624, 7, ['OverallQual', 'GrLivArea', 'GarageArea', 'p_TotSurf', 'PoolArea', 'p_TotalBath', 'YrSold']),
 (3748922.1862, 8, ['OverallQual', 'GarageArea', 'p_TotalBath', 'p_TotSurf', 'GrLivArea', 'PoolArea', 'YearRemodAdd', 'YearBuilt']),
 (3764732.48, 9, ['YearBuilt', 'GarageArea', 'PoolArea', 'YrSold', 'p_TotalBath', 'YearRemodAdd', 'GrLivArea', 'p_TotSurf', 'OverallQual']),
 (3766278.1881, 6, ['p_TotSurf', 'YearRemodAdd', 'p_TotalBath', 'OverallQual', 'GarageArea', 'YearBuilt']),
 (3773034.8089, 6, ['OverallQual', 'p_TotSurf', 'p_TotalBath', 'GarageArea', 'YearBuilt', 'YearRemodAdd']),
 (3777159.7975, 5, ['p_TotalBath', 'OverallQual', 'GarageArea', 'GrLivArea', 'p_TotSurf']),
 (3777707.8352, 7, ['p_TotSurf', 'GrLivArea', 'YrSold', 'GarageArea', 'OverallQual', 'YearRemodAdd', 'p_TotalBath']),
 (3778734.9465, 9, ['PoolArea', 'YearBuilt', 'YrSold', 'YearRemodAdd', 'p_TotalBath', 'GrLivArea', 'GarageArea', 'OverallQual', 'p_TotSurf']),
 (3779014.5223, 7, ['p_TotSurf', 'YearRemodAdd', 'OverallQual', 'GrLivArea', 'PoolArea', 'p_TotalBath', 'YearBuilt']),
 (3786737.1598, 8, ['p_TotSurf', 'YearBuilt', 'YrSold', 'OverallQual', 'p_TotalBath', 'YearRemodAdd', 'PoolArea', 'GarageArea']),
 (3789553.9813, 9, ['p_TotSurf', 'GrLivArea', 'YrSold', 'GarageArea', 'OverallQual', 'YearRemodAdd', 'p_TotalBath', 'YearBuilt', 'PoolArea']),
 (3792408.6662, 6, ['p_TotSurf', 'YearBuilt', 'PoolArea', 'GarageArea', 'OverallQual', 'p_TotalBath']),
 (3794035.5049, 6, ['YearRemodAdd', 'p_TotalBath', 'GrLivArea', 'GarageArea', 'OverallQual', 'p_TotSurf']),
 (3796698.4661, 6, ['OverallQual', 'YearBuilt', 'p_TotSurf', 'p_TotalBath', 'YearRemodAdd', 'GrLivArea']),
 (3799547.8999, 8, ['p_TotSurf', 'YearRemodAdd', 'p_TotalBath', 'OverallQual', 'GarageArea', 'YearBuilt', 'PoolArea', 'GrLivArea'])]





print("""
##########################################################
# 	9th Part :   Polynomial regression
##########################################################
""")


my_features = ['p_TotSurf', 'GrLivArea', 'YrSold', 'GarageArea', 'OverallQual', 
				'YearRemodAdd', 'p_TotalBath', 'YearBuilt']

sub_best_col_list = my_features
sub_X = X.loc[:, sub_best_col_list]

X_train, X_test, y_train, y_test = train_test_split(sub_X, y)






# plt.plot(y_test.index, y_test, label="test")


		# for deg in range(4) : 

		# 	polynomial_features = PolynomialFeatures(degree=deg, include_bias=True)
		# 	linear_regression = LinearRegression()

		# 	model = Pipeline([("polynomial_features", polynomial_features),
		# 						 ("linear_regression", linear_regression)])
		# 	model.fit(X_train, y_train)

		# 	y_pred = model.predict(X_test)
		# 	score = round(mean_squared_error(y_test, y_pred),4)**0.5
		# 	score_list.append(	(score, "LinearReressoin {} D".format(deg),
		# 						len(sub_best_col_list), sub_best_col_list))

		# 	plt.scatter(y_test, y_pred, marker = ".")
		# 	plt.title("LinearReressoin {} D".format(deg))

		# 	plt.show()
		# score_list.sort()

		# print(score_list[:20])


print("""
##########################################################
# 	10th Part :   Let's Go !!! 
##########################################################
""")



#  DF CRETAION

train_df = pd.read_csv("train.csv", index_col=0)
y_train = train_df["SalePrice"]
X_train  = train_df.drop(["SalePrice"], axis=1)


X_test = pd.read_csv("test.csv", index_col=0)



# CLEANING AND CREATING MY FEATURES

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
print(X_train.shape)
print(X_test.shape)
print(X_train.head())
print(X_test.head())



# DROP NA FROM X_TEST  !!! 

X_test["p_TotSurf"] = X_test["p_TotSurf"].fillna(X_test["p_TotSurf"].mean())
X_test["GarageArea"] = X_test["GarageArea"].fillna(X_test["GarageArea"].mean())
X_test["p_TotalBath"] = X_test["p_TotalBath"].fillna(X_test["p_TotalBath"].mean())


print("DF TRAIN")
na_col = 0
for col in X_train.columns : 
	na = X_train[col].isna().any()
	if na : 
		print("att {} has NaN!".format(col))
		na_col +=1
print("number of features with one/more NaN :", na_col)


print("DF test")
na_col = 0
for col in X_test.columns : 
	na = X_test[col].isna().sum()
	if na : 
		print("att {} has NaN! : {}".format(col, na))
		na_col +=1
print("number of features with one/more NaN :", na_col)



input("continuer?\n")



# regression

model = RidgeCV(normalize=True, cv=20)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(len(y_pred))

result = pd.DataFrame(dict(SalePrice=y_pred))
result.index += 1461
result.index.name = "ID"

result.to_csv("kaggle_submission.csv")


# my_features = ['p_TotSurf', 'GrLivArea', 'YrSold', 'GarageArea', 'OverallQual', 
# 				'YearRemodAdd', 'p_TotalBath', 'YearBuilt']

# sub_best_col_list = my_features
# sub_X = X.loc[:, sub_best_col_list]

# X_train, X_test, y_train, y_test = train_test_split(sub_X, y)


# # plt.plot(y_test.index, y_test, label="test")


# for deg in range(8) : 

# 	polynomial_features = PolynomialFeatures(degree=deg, include_bias=True)
# 	linear_regression = LinearRegression()

# 	model = Pipeline([("polynomial_features", polynomial_features),
# 						 ("linear_regression", linear_regression)])
# 	model.fit(X_train, y_train)

# 	y_pred = model.predict(X_test)
# 	score = round(mean_squared_error(y_test, y_pred),4)**0.5
# 	score_list.append(	(score, "LinearReressoin {} D".format(deg),
# 						len(sub_best_col_list), sub_best_col_list))

# 	plt.scatter(y_test, y_pred)
# 	plt.title("LinearReressoin {} D".format(deg))

# 	plt.show()
# score_list.sort()

# print(score_list[:20])