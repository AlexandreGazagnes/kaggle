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




############################################################################
#       Fihting against Missing / NaN values
############################################################################


#
#
#


# explore null/Nan values by col
total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


# Cabin, 77% is a very high score, we would have to drop this feature
# Embarked : 0.2% very very low effect on the dataset, we will choose to fill the Nan with the most common value of the feature
# Age, this isproblematic because age is an very important feature, and 20% is a prety high score
# we could, for instance, fill missing values with the mean age of each sub category sorting by fare, sex, survived and pclass ...

# what about the testing dataset?


# explore null/Nan values by col
total = test_df.isnull().sum().sort_values(ascending=False)
percent = (test_df.isnull().sum()/test_df.isnull().count()).sort_values(ascending=False)
pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


# we sould definitively delete "Cabin" ?
# we will use the same technique for Age and Fare than above

# just to be sure ...
data = train_df.loc[ : ,["Cabin", "Pclass", "Fare"]]
data.head(30).sort_values(by='Pclass', ascending=True)


# we could think that Pclass 3 is just a global common room or something like that, maybe we could confirm that with few web research but keep focus for now
# who are people without "Embarked" ?
train_df[train_df["Embarked"].isnull()]


# 1% come from Q, and for the rest 50/50 from C and S
# we will respect that proportion
train_df.loc[61, "Embarked"] = "C"
train_df.loc[829, "Embarked"] = "S"
train_df["Embarked"].isnull().all() == False


# ok, what about test_df
test_df[test_df["Fare"].isnull()]


# ouch we have a major issue ... 
# and for train_df
train_df[train_df["Fare"] == 0].sort_values(by="Pclass")


# re ouch ... what a mess 
# after few web resarch, we can leran than these people wer "sent by shipbuilders Harland & Wolff to accompany the Titanic on her maiden voyage." what a good gift ! :) :) 
# https://www.encyclopedia-titanica.org/titanic-guarantee-group/

# so is this missing value tricky, well not at all, we can fill missing value with average price of each class?

data = train_df.groupby("Pclass")
data = data.describe()["Fare"]
my_fare = (data["mean"] + data["50%"])/2
my_fare


# now we can fill Fare with my_fare
for c in [1,2,3] : 
	mask = (train_df["Fare"] == 0.0 ) & (train_df["Pclass"] == c)  
	train_df.loc[mask, "Fare"] = int(my_fare[c])
	

# just to be sure : 
train_df[train_df["Fare"] == 0.0]

train_df["Fare"].isnull().all() == False


# just to control : 
print(train_df.loc[302, ])
print(train_df.loc[263, ])


for c in [1,2,3] : 
	mask = ( (test_df["Fare"] == 0.0 ) | (test_df["Fare"].isnull()) ) & (test_df["Pclass"] == c)  
	test_df.loc[mask, "Fare"] = int(my_fare[c])



test_df[test_df["Fare"] == 0.0]
test_df["Fare"].isnull().all() == False


test_df[test_df["Fare"].isnull()]

test_df.loc[152, ]


# lets now deal with age,
# we will inpute random age rearding normal distribution
data = test_df.append(train_df.drop(["Survived"], axis=1), ignore_index=True)
data.head()

data = train_df[~train_df["Age"].isnull()]


data = train_df.loc[:, ["Age", "Pclass", "Sex"]].groupby(["Pclass", "Sex"])

data.describe()


# we define our mean and std for each subclass
coef = dict()
coef["female", 1] =  (37.037594, 14.272460)
coef["female", 2] = (27.499223,12.911747)
coef["female", 3] =  (22.185329, 12.205254)
coef["male", 1] = (41.029272, 14.578529)
coef["male", 2] = (30.815380,13.977400)
coef["male", 3] = (25.962264, 11.682415)



# we have to verify that each distribution is effectively normal : 

for cl in [1,2,3] : 
	age = train_df[~train_df["Age"].isnull()]
	age = age[age["Pclass"] == cl]
	sns.distplot(age["Age"], bins=50)


# good for class 

for sex in ["male", "female"] : 

	age = train_df[~train_df["Age"].isnull()]
	age = age[age["Sex"] == sex]
	sns.distplot(age["Age"], bins=50)



# good for sex

# we define our function to fill na
normal_dist = lambda x,mean, std : std * np.random.randn(x) + mean



# just a small print to check our values, and to test our  main loop

for sex in ["male", "female"] : 
    for cl in [1,2,3] : 
        mask = (train_df["Sex"] == sex) & (train_df["Pclass"] == cl) & (train_df["Age"].isnull())
        nb = len(train_df[mask])
        print(str(nb))
        print(train_df.loc[mask, ["Sex", "Pclass", "Age"]])



# we fill na

for sex in ["male", "female"] : 
    for cl in [1,2,3] : 
        mask = (train_df["Sex"] == sex) & (train_df["Pclass"] == cl) & (train_df["Age"].isnull())
        nb = len(train_df[mask])
        train_df.loc[mask, "Age"] = normal_dist(nb, coef[sex, cl][0], coef[sex, cl][1])



train_df["Age"].isnull().all() == False
train_df[train_df["Age"].isnull()]

test_df[test_df["Age"].isnull()]



for sex in ["male", "female"] : 
    for cl in [1,2,3] : 
        mask = (test_df["Sex"] == sex) & (test_df["Pclass"] == cl) & (test_df["Age"].isnull())
        nb = len(test_df[mask])
        test_df.loc[mask, "Age"] = normal_dist(nb, coef[sex, cl][0], coef[sex, cl][1])




test_df[test_df["Age"].isnull()]



# OK we have made a GREAT step !!! 
# we have no Nan/null values, 
# we were very prudent in filling na in order not to overfit or underfit in our futuress manipulations

# ONE GOOD THING TO DO COULD BE TO FILL FARE WITH THE SAME METHOD BUT IS IT VERY USEFULL? I DON'T KNOW'



############################################################################
#       Feature Engineering
############################################################################


# thirst we will convert all cat fearurs in int : 

# train_df["Sex"] = train_df["Sex"].apply (lambda x : 1 if x == "male" else 0)
# test_df["Sex"] = test_df["Sex"].apply (lambda x : 1 if x == "male" else 0)

# pairs = [("C", 0), ("S", 1), ("Q", 2)]
# train_df["Embarked"] = train_df["Embarked"].apply(lambda x :  [p[1] for p in pairs if x == p[0]][0])
# test_df["Embarked"] = test_df["Embarked"].apply(lambda x :  [p[1] for p in pairs if x == p[0]][0])


train_df["Pclass"] = train_df["Pclass"].apply(lambda x : str(x))
test_df["Pclass"] = test_df["Pclass"].apply(lambda x : str(x))

train_df["Age"] = train_df["Age"].astype(dtype="int32")
test_df["Age"] = test_df["Age"].astype(dtype="int32")

train_df["Fare"] = train_df["Fare"].astype(dtype="int32")
test_df["Fare"] = test_df["Fare"].astype(dtype="int32")

train_df["Parch"] = train_df["Parch"].astype(dtype="int32")
test_df["Parch"] = test_df["Parch"].astype(dtype="int32")

train_df["SibSp"] = train_df["SibSp"].astype(dtype="int32")
test_df["SibSp"] = test_df["SibSp"].astype(dtype="int32")

# then we can create a new features 
train_df["Alone"] = pd.Series(train_df["Parch"]  + train_df["SibSp"], name="Alone")
train_df["Alone"] = train_df["Alone"].apply(lambda x : 1 if not x else 0)
test_df["Alone"] = pd.Series(test_df["Parch"]  + test_df["SibSp"], name="Alone")
test_df["Alone"] = test_df["Alone"].apply(lambda x : 1 if not x else 0)



train_df["Parch"] = train_df["Parch"].apply(lambda x : 0 if x == 0 else 1)
test_df["Parch"] = test_df["Parch"].apply(lambda x : 0 if x == 0 else 1)

train_df["SibSp"] = train_df["SibSp"].apply(lambda x : 0 if x == 0 else 1)
test_df["SibSp"] = test_df["SibSp"].apply(lambda x : 0 if x == 0 else 1)



# we drop useless (for now) features
train_df = train_df.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1)
test_df = test_df.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1)

# We transorm Age and Fare and Age in categrocical features


def shape_age(x) : 
	if x <=3 : 
		return 10 
	if  3< x <=7 : 
		return 10
	if  7< x <=10 : 
		return 10
	if  10< x <=14 : 
		return 8
	if  14< x <=20 : 
		return 6
	if  20< x <=30 : 
		return 4
	if  30< x <=40 : 
		return 2
	if  40< x <=50 : 
		return 2
	if  50< x <=60 : 
		return 4
	if  60< x <=70 : 
		return 8
	return 10

train_df["Age"] = train_df["Age"].apply(shape_age)
train_df["Age"].astype("int16")
test_df["Age"] = test_df["Age"].apply(shape_age)
test_df["Age"].astype("int16")

train_df["Fare"] = pd.cut(train_df["Fare"], 10, labels=range(10))
train_df["Fare"].astype("int16")

test_df["Fare"] = pd.cut(test_df["Fare"], 10, labels=range(10))
test_df["Fare"].astype("int16")

# train_df = train_df.drop(["Fare"], axis=1)
# test_df = test_df.drop(["Fare"], axis=1)


train_df.head()




train_df = pd.get_dummies(train_df)
test_df = pd.get_dummies(test_df)
train_df.head()



X_train, X_test, y_train, y_test = train_test_split(train_df.drop(["Survived"], axis = 1), train_df.loc[:,"Survived"])

X_train.shape
X_test.shape
y_train.shape
y_test.shape




############################################################################
#      First step Random Forest
############################################################################

# rf = RandomForestClassifier()
# params = {"n_jobs" : [300, 500, 700, 1000], 
# 			"oob_score" : [True, False], 
# 			"bootstrap":[True,], # False	 
# 			"warm_start" : [True, False],
# 			"max_features" : ["auto", "log2"]}

# best_params = {'warm_start': [True], 'oob_score': [True], 'bootstrap':[ True,] ,
# 				'max_features': ['auto'], 'n_jobs': [300, 500, 700, 900, 1000]}



# grf = GridSearchCV(rf, best_params, cv= 10, scoring="accuracy")
# grf.fit(X_train, y_train)
# y_pred = grf.predict(X_test)
# score = grf.score(X_test, y_test)
# print(score)
# print(grf.best_params_)
# print(grf.best_estimator_)

# errors = y_pred + y_test
# errors_rate = len([i for i in errors if i ==1]) /len(y_pred)
# print(errors_rate)

# print(score+errors_rate)


# y_sub = grf.predict(test_df)


# y_sub


# sub = pd.concat([pd.Series(range(892, 892 + len(y_sub)), name="PassengerId"), pd.Series(y_sub, name="Survived")], axis=1)

# sub.to_csv("submission.csv", index=False)





gbc = GradientBoostingClassifier()
params = {	"n_estimators" : 	[10, 25, 50, 75, 100], 
			"learning_rate" : 	np.logspace(-3, 3, 6),
			"loss" : 			["deviance", "exponential"],
			"max_features" : 	["auto", "log2"],
			"warm_start" : [True, False]}



ggbc = GridSearchCV(gbc, params, cv= 10, scoring="accuracy")
ggbc.fit(X_train, y_train)
y_pred = ggbc.predict(X_test)
score = ggbc.score(X_test, y_test)
print(score)
print(ggbc.best_params_)
print(ggbc.best_estimator_)

errors = y_pred + y_test
errors_rate = len([i for i in errors if i ==1]) /len(y_pred)
print(errors_rate)

print(score+errors_rate)

