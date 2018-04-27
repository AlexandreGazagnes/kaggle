#!/usr/bin/env python3
# -*- coding: utf-8 -*-



############################################
# 	House price competition
############################################


# Find here my work regarding my dirst kaggle competition
# This is a kernel, not a script, SPECIALIZED in LASSO technique
# found here all perparatory work, studies and model selection

# based on House price dataset


###############################################
# 	Import and configs 
###############################################


#  calulation packages 

import numpy as np
import pandas as pd
from scipy import stats


# visualisation tools

import matplotlib.pyplot as plt
# if iptyhon/jupyter  	%matplotlib inline
import seaborn as sns
sns.set()


# skearln modules
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV, Ridge, Lasso,\
								ElasticNet, LassoCV, LassoLarsCV, HuberRegressor
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV






##########################################################
# 	Global variables and constants
##########################################################


SHOW = False # if True all plt.show()  / print will be activated not 
# see if SHOW  : ... for more info



##########################################################
# 	Personnal global functions
##########################################################


def print_t(title):
	"""just a fancy personal print"""

	if title : 
		print("\n##########################################################")
		print("\t" + title)
		print("##########################################################\n")

def print_e(): 
	print()
	print()

def pause() : 
	print("\n\n")
	print_t("continuer?") 
	input()




##########################################################
# 	DataFrame creation 
##########################################################


df_train = pd.read_csv("train.csv")
df_train.drop(["Id"], axis=1, inplace=True)



##########################################################
# 	First data exploration
##########################################################


# explore dataframe columns

if SHOW : 	
	print_t("DataFrame info")
	print(df_train.info())
	print_e()
	print(df_train.describe())
	print_e()
	print(df_train.dtypes)
	print_e()
	print(df_train.shape)
	print_e()
	print(df_train.ndim)
	print_e()

if SHOW : 	
	print_t("DataFrame columns")
	print(df_train.columns)
	print_e()


# explore Saleprice main properties

if SHOW : 
	print_t("SalePrice feature")
	print(df_train['SalePrice'].describe())
	print_e()

	print("Skewness: {}".format(df_train['SalePrice'].skew()))
	print("Kurtosis: {}".format(df_train['SalePrice'].kurt()))
	print_e()

if SHOW : 
	sns.distplot(df_train['SalePrice'])
	plt.show()


# explore main corelation between variables

if SHOW :
	fig, ax = plt.subplots(1,2, figsize=(10,5))
	for i, var in enumerate(['GrLivArea', 'TotalBsmtSF' ]) : 
		ax[i].scatter(x=df_train[var], y=df_train['SalePrice'], marker=".")
		ax[i].set_title("{} vs SalePrice".format(var))
		plt.show()


# explore corelation for categorical values

if SHOW :
	var = 'OverallQual'
	data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
	f, ax = plt.subplots(figsize=(8, 6))
	fig = sns.boxplot(x=var, y="SalePrice", data=data)
	fig.axis(ymin=0, ymax=800000)
	plt.show()

if SHOW :
	var = 'YearBuilt'
	data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
	f, ax = plt.subplots(figsize=(16, 8))
	fig = sns.boxplot(x=var, y="SalePrice", data=data)
	fig.axis(ymin=0, ymax=800000)
	plt.xticks(rotation=90)
	plt.show()

if SHOW :
	var = 'YearRemodAdd'
	data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
	f, ax = plt.subplots(figsize=(16, 8))
	fig = sns.boxplot(x=var, y="SalePrice", data=data)
	fig.axis(ymin=0, ymax=800000)
	plt.xticks(rotation=90)
	plt.show()


#correlation matrix of all features

if  SHOW:
	corrmat = df_train.corr()
	f, ax = plt.subplots(figsize=(12, 9))
	sns.heatmap(corrmat, vmax=.9, square=True)
	plt.show()


# correlation maxtix of "important" features

if  SHOW :
	k = 10 #number of variables for heatmap
	cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
	cm = np.corrcoef(df_train[cols].values.T)
	sns.set(font_scale=1.25)
	hm = sns.heatmap(	cm, cbar=True, annot=True, square=True, fmt='.2f', 
						annot_kws={'size': 10}, 
						yticklabels=cols.values, xticklabels=cols.values)
	plt.show()


# scatterplot all important features

if SHOW :
	sns.set()
	important_features = cols = ['SalePrice', 'OverallQual', 'GrLivArea', 
								'GarageCars', 'TotalBsmtSF', 'FullBath', 
								'YearBuilt']
	sns.pairplot(df_train[cols], size = 2.5)
	plt.show()




##########################################################
# 	Data cleaning : missing values  - NaN
##########################################################


if SHOW : 		
	na_col = 0
	for col in df_train.columns : 
		na = df_train[col].isna().all()
		if na : 
			print("att {} has only NaN!".format(col))
			na_col +=1
	print("number of features with one/more NaN :", na_col)


# delete rows with all NaN values

df_train.dropna(axis=1, how="all", inplace=True)
df_train.dropna(axis=0, how="all", inplace=True)


if SHOW : 		
	na_col = 0
	for col in df_train.columns : 
		na = df_train[col].isna().any()
		if na : 
			print("att {} has NaN!".format(col))
			na_col +=1
	print("number of features with one/more NaN :", na_col)


# drop if any NaN

df_train.dropna(axis=1, how='any', inplace=True)
df_train.dropna(axis=0, how='any', inplace=True)



##########################################################
# 	Data cleaning : missing values  -  Null
##########################################################

# any Null?


if SHOW : 
	null_col = 0
	for col in df.columns : 
		null = df[col].isnull().all()
		if null : 
			print("att {} has only Null values ? : {}".format(col, null))
			null_col +=1
	print("number of features with only null :", null_col)




# finding missing values

def depreciated_missing_values_finding() : 
	# deprecated :
	null = [	(col,df_train[col].isnull().sum()) for col in df_train.columns \
				if df_train[col].isnull().sum() ]
	null.sort(key=lambda i : i[1], reverse=True)
	[print("{} has {} null".format(col, null_val )) for col, null_val in null]
	print_e()


# pythonic way of life :)
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


if SHOW : 
	print_t("Missing data")
	print(missing_data.head(20))
	print_e()


# deleting missing data
try : 
	df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
	df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
except : 
	print("value already deleted	")


##########################################################
# 	Data cleaning :  outliers
##########################################################


# unvariate analysis with standardizing data

saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]



if SHOW : 
	print_t("Outliers, unvariate analysis")
	print('outer range (low) of the distribution:')
	print(low_range)
	print('\nouter range (high) of the distribution:')
	print(high_range)
	print_e()


if SHOW : 
	print_t("Outliers, bivariate analysis")
	var = 'GrLivArea'
	data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
	data.plot.scatter(	x=var, y='SalePrice', ylim=(0,800000), marker=".", 
						xticks=range(0, 7000, 300))
	plt.show()


# lest find features best corelated with sale Price

corrmat = df_train.corr()
corrmat = corrmat.where(corrmat !=1, np.nan)
corrmat = corrmat.loc[ :, "SalePrice"]
corrmat = corrmat.sort_values(ascending=False)
corrmat = corrmat[corrmat >0.5]
corr_mat_list = list(corrmat.index) 


if SHOW : 

	for col in corr_mat_list : 
		data = pd.concat([df_train['SalePrice'], df_train[col]], axis=1)
		data.plot.scatter(	x=col, y='SalePrice', marker=".")	
		plt.show()
		print("{}".format(col))
	pause()


# deleting outliers 

df_train = df_train[df_train["GrLivArea"]  < 4550]
df_train = df_train[df_train["TotalBsmtSF"]  < 3000]
df_train = df_train.drop(["GarageCars"], axis=1)
df_train = df_train[df_train["GarageArea"]  < 1200]
df_train = df_train.drop("TotRmsAbvGrd", axis=1)

[corr_mat_list.remove(val) for val in  ("GarageCars", "TotRmsAbvGrd")]





##########################################################
# 	Data cleaning :  Rescaling data
##########################################################


# focus on best features 

if SHOW : 
	data = df_train.loc[:, corr_mat_list]
	data.hist(bins=50)
	plt.show()

best_features_skew_and_kur = [	(col, 
								round(df_train[col].skew(),2), 
								round(df_train[col].kurt(),2))\
									 for col in corr_mat_list]


best_features_skew_and_kur.sort(key=lambda v : v[1], reverse=True)

if not SHOW : 
	print_t("Skew and Kurt for 'Best features'")
	[print("feature {}, skew {}, kut {}".format(i,j,k)) \
		for i,j,k in best_features_skew_and_kur]
	print_e()


if  SHOW : 
	for feature in best_features_skew_and_kur[:2] : 		
		sns.distplot(df_train[feature[0]], fit=norm)
		plt.show()
		stats.probplot(df_train[feature[0]], plot=plt)
		plt.show()


		# #  rescale our 2 first features
		# for feature in [i[0] for i in best_features_skew_and_kur[:2]] : 
		# 	df_train[feature] = np.log(df_train[feature])


		# if  SHOW : 
		# 	for feature in best_features_skew_and_kur[:2] : 		
		# 		sns.distplot(df_train[feature[0]], fit=norm)
		# 		plt.show()
		# 		stats.probplot(df_train[feature[0]], plot=plt)
		# 		plt.show()


		# def depreciated_rescale_SalePrice() : 
		# 	# before rescale 
		# 	if  SHOW: 
		# 		sns.distplot(df_train['SalePrice'], fit=norm)
		# 		plt.show()
		# 		stats.probplot(df_train['SalePrice'], plot=plt)
		# 		plt.show()

		# 	# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		# 	# if +/ normal distribution and skeansess > 1 --> LOG TRANSFORM
		# 	# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		# 	df_train['SalePrice'] = np.log(df_train['SalePrice'])

		# 	# after rescale
		# 	if  SHOW:
		# 		sns.distplot(df_train['SalePrice'], fit=norm);
		# 		fig = plt.figure()
		# 		res = stats.probplot(df_train['SalePrice'], plot=plt)
		# 		plt.show()




##########################################################
# 	Data cleaning : Cateorcial / Dummy features
##########################################################


def depreciated_get_dummies() : 
	df_train = pd.get_dummies(df_train)




##########################################################
# 	Data cleaning : creating personnal features
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
		df_train = df_train.drop([feature], axis=1)
	except : 
		print("{} already deleted :) ".format(feature))

df = df_train


numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns.tolist()]
cat_cols = [col for col in df.select_dtypes(exclude=[np.number]).columns.tolist()]

df.drop(cat_cols, axis=1, inplace=True)



##########################################################
# 	Standardize and Split  
##########################################################


X = df.drop(["SalePrice"], axis=1)
y = df["SalePrice"]

X  = StandardScaler().fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


##########################################################
# 	Linear Regression 
##########################################################



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



##########################################################
# 	Ridge Regression 
##########################################################


n_alphas = 100
alphas = np.logspace(-5, 5, 100 )

coefs = list()
errors = list()

for a in alphas : 
	ridge = Ridge(alpha=a)
	clf = GridSearchCV(ridge,{}, cv=20)
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




##########################################################
# 	Lasso Regression 
##########################################################


n_alphas = 100
alphas = np.logspace(-5, 5, 100 )

coefs = list()
errors = list()

for a in alphas : 
	lasso = Lasso(alpha=a)
	lasso.fit(X_train, y_train)
	coefs.append(lasso.coef_)
	y_pred = lasso.predict(X_test)
	errors.append(round(rmsle(y_test, y_pred),4))  




plt.plot(alphas, errors)
plt.plot(alphas, [baseline_error for _ in alphas])
plt.xscale('log')
plt.ylim([0.1, 0.3])
plt.show()


errors = list(zip(alphas, errors))
errors.sort(key=lambda x : x[1])
print(errors[:10])	



##########################################################
# 	Saving results
##########################################################


result = pd.DataFrame(dict(SalePrice=y_pred))
result.index += 1461 ; result.index.name = "ID"

result.to_csv("kaggle_submission.csv")

