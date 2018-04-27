#!/usr/bin/env python3
# -*- coding: utf-8 -*-



############################################
# 	House price competition
############################################



# Find here my work regarding my dirst kaggle competition
# This is a kernel, not a script, 
# found here all perparatory work, studies and model selection
# before running a script and submitings predictions

# v1.0 just data analysis, exploration and first data cleaning ops
# no model tests, no advance reression techniques



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



##########################################################
# 	Global variables and constants
##########################################################


PLT = True # if True all plt.show() will be activated not 
# see if PLT : ... for more info

PRT = True # if True all print() will be activated not 
# see if PRT : ... for more info

sns.set()


##########################################################
# 	Personnal global functions
##########################################################


def print_t(title):
	"""just a fancy personal print"""

	if title : 
		print("##########################################################")
		print("\t" + title)
		print("##########################################################\n")

def print_e(): 
	print()
	print()



##########################################################
# 	DataFrame creation 
##########################################################


df_train = pd.read_csv("train.csv")



##########################################################
# 	First data exploration
##########################################################


# explore dataframe columns

if PRT : 	
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

if PRT : 	
	print_t("DataFrame columns")
	print(df_train.columns)
	print_e()


# explore Saleprice main properties

if PRT : 
	print_t("SalePrice feature")
	print(df_train['SalePrice'].describe())
	print_e()

	print("Skewness: {}".format(df_train['SalePrice'].skew()))
	print("Kurtosis: {}".format(df_train['SalePrice'].kurt()))
	print_e()

if PLT : 
	sns.distplot(df_train['SalePrice'])
	plt.show()


# explore main corelation between variables

if PLT :
	fig, ax = plt.subplots(1,2, figsize=(10,5))
	for i, var in enumerate(['GrLivArea', 'TotalBsmtSF' ]) : 
		ax[i].scatter(x=df_train[var], y=df_train['SalePrice'], marker=".")
		ax[i].set_title("{} vs SalePrice".format(var))
		plt.show()


# explore corelation for categorical values

if PLT :
	var = 'OverallQual'
	data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
	f, ax = plt.subplots(figsize=(8, 6))
	fig = sns.boxplot(x=var, y="SalePrice", data=data)
	fig.axis(ymin=0, ymax=800000)
	plt.show()

if PLT :
	var = 'YearBuilt'
	data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
	f, ax = plt.subplots(figsize=(16, 8))
	fig = sns.boxplot(x=var, y="SalePrice", data=data)
	fig.axis(ymin=0, ymax=800000)
	plt.xticks(rotation=90)
	plt.show()

if PLT :
	var = 'YearRemodAdd'
	data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
	f, ax = plt.subplots(figsize=(16, 8))
	fig = sns.boxplot(x=var, y="SalePrice", data=data)
	fig.axis(ymin=0, ymax=800000)
	plt.xticks(rotation=90)
	plt.show()


#correlation matrix of all features

if PLT :
	corrmat = df_train.corr()
	input("continuer?\t")
	f, ax = plt.subplots(figsize=(12, 9))
	sns.heatmap(corrmat, vmax=.8, square=True)
	plt.show()


# correlation maxtix of "important" features

if PLT :
	k = 10 #number of variables for heatmap
	cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
	cm = np.corrcoef(df_train[cols].values.T)
	sns.set(font_scale=1.25)
	hm = sns.heatmap(	cm, cbar=True, annot=True, square=True, fmt='.2f', 
						annot_kws={'size': 10}, 
						yticklabels=cols.values, xticklabels=cols.values)
	plt.show()


# scatterplot all important features

if PLT :
	sns.set()
	important_features = cols = ['SalePrice', 'OverallQual', 'GrLivArea', 
								'GarageCars', 'TotalBsmtSF', 'FullBath', 
								'YearBuilt']
	sns.pairplot(df_train[cols], size = 2.5)
	plt.show()



##########################################################
# 	Data cleaning : missing values  - Nan and Null
##########################################################


# finding missing values

		# deprecated :

		# null = [	(col,df_train[col].isnull().sum()) for col in df_train.columns \
		# 			if df_train[col].isnull().sum() ]
		# null.sort(key=lambda i : i[1], reverse=True)
		# [print("{} has {} null".format(col, null_val )) for col, null_val in null]
		# print_e()


# pythonic way of life :)
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

if PRT : 
	print_t("Missing data")
	print(missing_data.head(20))
	print_e()


# deleting missing data

df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)



##########################################################
# 	Data cleaning :  outliers
##########################################################


# unvariate analysis with standardizing data

saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

if PRT : 
	print_t("Outliers, unvariate analysis")
	print('outer range (low) of the distribution:')
	print(low_range)
	print('\nouter range (high) of the distribution:')
	print(high_range)
	print_e()



# bivariate analysis with standardizing data

saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

if PLT : 
	print_t("Outliers, bivariate analysis")
	var = 'GrLivArea'
	data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
	data.plot.scatter(	x=var, y='SalePrice', ylim=(0,800000), marker=".", 
						xticks=range(0, 7000, 300))
	plt.show()


# deleting outliers 

df_train =df_train[df_train["GrLivArea"]  < 4550]



##########################################################
# 	Data cleaning :  Rescaling data
##########################################################


		# # about SalePrice

		# # before rescale 
		# if not PLT : 
		# 	sns.distplot(df_train['SalePrice'], fit=norm)
		# 	plt.show()
		# 	stats.probplot(df_train['SalePrice'], plot=plt)
		# 	plt.show()

		# # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		# # if +/ normal distribution and skeansess > 1 --> LOG TRANSFORM
		# # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		# df_train['SalePrice'] = np.log(df_train['SalePrice'])

		# # after rescale
		# if not PLT :
		# 	sns.distplot(df_train['SalePrice'], fit=norm);
		# 	fig = plt.figure()
		# 	res = stats.probplot(df_train['SalePrice'], plot=plt)
		# 	plt.show()


# about GrLivArea 

# before rescale 
if PLT : 
	sns.distplot(df_train['GrLivArea'], fit=norm)
	plt.show()
	stats.probplot(df_train['GrLivArea'], plot=plt)
	plt.show()

#rescale
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])

# after rescale
if PLT :
	sns.distplot(df_train['GrLivArea'], fit=norm);
	fig = plt.figure()
	res = stats.probplot(df_train['GrLivArea'], plot=plt)
	plt.show()



##########################################################
# 	Data cleaning : Cateorcial / Dummy features
##########################################################


		# df_train = pd.get_dummies(df_train)

