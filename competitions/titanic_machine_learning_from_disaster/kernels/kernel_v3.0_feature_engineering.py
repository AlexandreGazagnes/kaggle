#!/usr/bin/env python3
# -*- coding: utf-8 -*-



################################################################################
#       Titanic : Feature Engineering 
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




# Missing / NaN values
############################################################################


# In this First part, we will explore missin and NanN values
# We will trying to spot what features are missing, and their respicve impact 
# regarding dataset quality


# first "classic" method
print(train_df.isnull().all())
print(train_df.isnull().any())


# explore null/Nan values by col
total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
print(pd.concat([total, round(100 * percent,2)], axis=1, keys=['Total', 'Percent']))


# Cabin, 77% is a very high score, maybe we would have to drop this feature
# Embarked : 0.2% very very low effect on the dataset, we will choose to fill 
# the Nan with the most common value of the feature
# Age, this is a very problematic topic because age is an seems to be a very 
# important feature, and 20% is a prety high score
# we could, for instance, fill missing values with the mean age of each sub 
#category sorting by fare, sex, survived and pclass ...

# what about the testing dataset?


# first "classic" method
print(test_df.isnull().all())
print(test_df.isnull().any())


# explore null/Nan values by col
total = test_df.isnull().sum().sort_values(ascending=False)
percent = (test_df.isnull().sum()/test_df.isnull().count()).sort_values(ascending=False)
print(pd.concat([total, round(100 * percent,2)], axis=1, keys=['Total', 'Percent']))



# Cabin
############################################################################

# we sould definitively delete "Cabin" ?
# we will use the same technique for Age and Fare than above

# just to be sure ...
data = train_df.loc[ : ,["Cabin", "Pclass", "Fare"]]
print(data.head(30).sort_values(by='Pclass', ascending=True))

# we could think that Pclass 3 is just a global common room or something like
# that, maybe we could confirm that with few web research but keep focus for now



# Emabrked
############################################################################

# who are people without "Embarked" ?
print(train_df[train_df["Embarked"].isnull()])


# with Pclass
data = train_df[train_df['Pclass'] == 1]
data = pd.DataFrame({"count":data["Embarked"].value_counts()})
data["percent"] = round(100 * data["count"] / data["count"].sum(), 2)
print(data)

# without Pclass
data = pd.DataFrame({"count":train_df["Embarked"].value_counts()})
data["percent"] = round(100 * data["count"] / data["count"].sum(), 2)
print(data)


# First idea : Prob says "S"
# WWho are  they? "Icard Miss amelie and Stone, Mrs George Nelson Evevlyn"
# we could suppose it sound more "English" than "French"
# same ticket, same cabin, ok same Embarked value
# but ... why not from Ireland? 
# let's check this

data = train_df[train_df['Embarked'] == "Q"]
data = pd.DataFrame({"count":data["Pclass"].value_counts()})
data["percent"] = round(100 * data["count"] / data["count"].sum(), 2)
print(data)

# Wow Wow Wow 93% of people comming from Ireland were in Pclass 3 ? 
# such an info regarding massive emigration from irland to the US :) 

train_df.loc[61, "Embarked"] = "S"
train_df.loc[829, "Embarked"] = "S"
print(train_df["Embarked"].notnull().all())
print(test_df["Embarked"].notnull().all())



# Fare
############################################################################


# ok, what about test_df Fare
print(test_df[test_df["Fare"].isnull()])

# ok, is null  means NAN, but everybody paid for their ticket?
print(test_df[test_df["Fare"] == 0].sort_values(by="Pclass"))


# ouch we have a major issue ... 
# and for train_df
print(train_df[train_df["Fare"] == 0].sort_values(by="Pclass"))



# re ouch ... what a mess 
# after few web resarch, we can leran than these people wer "sent by shipbuilders Harland & Wolff to accompany the Titanic on her maiden voyage." what a good gift ! :) :) 
# https://www.encyclopedia-titanica.org/titanic-guarantee-group/

# so is this missing value tricky, well not at all, we can fill missing value with average price of each class?

data = train_df.groupby("Pclass")
data = data.describe()["Fare"]
my_fare = (data["mean"] + data["50%"])/2
print(my_fare)


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


# Age
############################################################################


# Strategy 1 : Grouping age from test_df and train_df


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



########################################################################
raise ValueError("\n\n{0}\n\tPause\n{0}\n\n".format(40*"*"))
########################################################################


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

train_df = train_df.drop(["Fare"], axis=1)
test_df = test_df.drop(["Fare"], axis=1)


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




# def meta params
gbc = GradientBoostingClassifier()	
params = {	"n_estimators" : 	[10, 25, 50, 75, 100, 200], 
			"learning_rate" : 	np.logspace(-3, 2, 5),
			"loss" : 			["deviance", "exponential"],
			"max_features" : 	["auto", "log2"],
			"warm_start" : 		[True, False]}


best_params = {'max_features': ['auto'], 'loss': ['deviance'], 'n_estimators': [50], 'warm_start': [True], 'learning_rate': [0.01]}


# launch and fit
ggbc = GridSearchCV(gbc, best_params, cv= 10, scoring="accuracy")
ggbc.fit(X_train, y_train)


# print results
score = ggbc.score(X_test, y_test)
print(score)
print(ggbc.best_params_)
print(ggbc.best_estimator_)


# just to be sure :) :) 
y_pred = ggbc.predict(X_test)
errors = y_pred + y_test
errors_rate = len([i for i in errors if i ==1]) /len(y_pred)
print(errors_rate)
print(score+errors_rate)


# make pred
y_sub = ggbc.predict(test_df)

# submit pred
sub = pd.concat([pd.Series(range(892, 892 + len(y_sub)), name="PassengerId"), pd.Series(y_sub, name="Survived")], axis=1)

sub.to_csv("submission.csv", index=False)
