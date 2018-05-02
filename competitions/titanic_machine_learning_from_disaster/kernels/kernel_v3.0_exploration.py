#!/usr/bin/env python3
# -*- coding: utf-8 -*-



################################################################################
#       Titanic : Full Intro/Tutorial for Data Science ?
################################################################################



# Hello and Welcome on bord! 
# This is my first kernel, and my first Kaggle competition. This kernel is 
# written for newbies, many, many lines could be factorized but I decided to write the simplest and more readable code I could
# # Of course if you want to propose to improve anything in the lines bellow, 
# feel free to post a comment!



# import

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
# %matplotlib inline

import seaborn as sns
sns.set()

from sklearn.preprocessing import * 
from sklearn.model_selection import *
from sklearn.linear_model import * 
from sklearn.metrics import r2_score
from sklearn.ensemble import *



#  dataframe creation

train_df = pd.read_csv("../datasets/train.csv")
test_df = pd.read_csv("../datasets/test.csv")



# Constant 

SHOW = True



############################################################################
#        First round of exploration 
############################################################################


# in this part we will print and buid graph with a brutal approch. we won't try to have a complex analysis but just to 
# have a first and global vision of our dataset ! 


# let's print main info about train_df
train_df.head()


train_df.describe()

print(train_df.ndim)
print(train_df.shape)

test_df.head()
test_df.describe()
print(test_df.ndim)
print(test_df.shape)


# less rows in test_df, logical
# 1 feature missing in test_df, logical : our target


# from now we will focus on train_df

pd.DataFrame(train_df.dtypes, columns=["type"])


# how many non unique values for each feature?
data = pd.DataFrame([len(train_df[feat].unique()) for feat in train_df.columns], columns=["unique values"], index=train_df.columns)
data.sort_values(by="unique values", ascending=False)


# Ok, nb of different values for PassengerID, Name, Ticket == nb of passenger, logical
# Sex, Survived, Emarked, Parch and SibSp looks like Categorcial features (?) to be confirmed...
# We will need to check Fare, Cabin, Tiket and Age...

# let's have a brutal distribution plot
_ = train_df.hist(grid=True,bins=50, figsize=(10,11))


# ok, Fare, Parch, SibSp seems to have a log distribution fonction 
# Age +/- normal with positive skewness and kurosis (to be confirmed)
# Pclass and Survied definitively are categorical features

# abrutal corelation matrix ? let's go! 
corrmat = train_df.corr()
f, ax = plt.subplots(figsize=(12, 9))
_ = sns.heatmap(corrmat, vmax=.8, square=True, fmt='.2f', annot=True)


# Pclass has strong negative correlation with price, logical
# Survied has good correlation with Fare and Pclass, such a capitalist world 100 years before !:) 
# how about surprising relations? 
	# Age and Pclass? No ... compare people's age in a concert and in a opera for fun...
	# Parch / 1ge /SibSp ... seems to be logical but could be studied more deeply later
# Ok so noting outsanding

# as we said before, Pclass, Sex, and Embarqued are categorcial features, so let's work on this
for feat in ["Embarked", "Sex", "Pclass"] : 
	sns.factorplot(feat,'Survived', data=train_df,size=4,aspect=3)


# Ok good, very good, we have very important features with very strong correlations to our target
# Just for fun we could compute the surviving rate of a woman in 1st class, 40+ years old, embarked at Cherbourg, but later :) 

# Can we learn something treating our continuous features as categrocial ones ? 
for feat in ["Age", "Fare", ] : 
	dat = pd.concat([pd.cut(train_df[feat], 10), train_df["Survived"]], axis=1)
	dat.columns = [feat, "Survived"]
	sns.factorplot(feat, "Survived", data=dat, size=4,aspect=3)


# Humm not very good ! 
# Seaborn is our freind? For sure! 

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))
sns.countplot(x='Embarked', data=train_df, ax=axis1)
sns.countplot(x='Survived', hue="Embarked", data=train_df, order=[1,0], ax=axis2)

embark_perc = train_df[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
_ = sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)


# average survived passengers by age
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
data = train_df[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
_ = sns.barplot(x="Age", y='Survived', data=data)



# average survived passengers by Fare
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
data = train_df.loc[:, ["Fare", "Survived", "Sex"]]
data["cat_fare"] = pd.cut(data["Fare"], 100, labels=range(100))
data = data[["cat_fare", "Survived"]].groupby(['cat_fare'],as_index=False).mean()
_ = sns.barplot(x='cat_fare', y='Survived', data=data)


