#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 13:20:12 2020

@author: riju
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

titanic_data=pd.read_csv('/Users/riju/Desktop/IIM_Calcutta_Study/IIMC_Study/EXTRAS !/CPBA/analytics/Projects/pyanalytics/Titanic.csv')

titanic_data.columns

titanic=titanic_data

titanic

#data analysis

len(titanic['PassengerId'])

sns.countplot(x="Survived",data=titanic)

sns.countplot(x="Survived",hue="Sex",data=titanic)

sns.countplot(x="Survived",hue="Pclass",data=titanic)

titanic['Fare'].plot.hist(bins=20,figsize=(7,5))

titanic.info()

sns.countplot(x="SibSp",hue="Survived",data=titanic)

#data wrangling

titanic.isnull()

titanic.isnull().sum()

sns.heatmap(titanic.isnull(),yticklabels='False')

sns.boxplot(x="Pclass",y="Age",data=titanic)

titanic.drop("Cabin",axis=1,inplace=True)

titanic.head()

sns.heatmap(titanic.isnull(),yticklabels='False',cbar=False)

titanic.dropna(inplace=True)

titanic.isnull().sum()

#pd.get_dummies(titanic["Sex"],drop_first='True')
sex=pd.get_dummies(titanic["Sex"],drop_first=True)

sex.head()

embark=pd.get_dummies(titanic["Embarked"],drop_first=True)

embark.head()

Pcl=pd.get_dummies(titanic["Pclass"],drop_first=True)
Pcl.head()

titanic=pd.concat([titanic,sex,embark,Pcl],axis=1)

titanic.head()

titanic.drop(['Sex','Embarked','PassengerId','Name','Ticket'],axis=1,inplace=True)

titanic.head()

titanic.drop(['Pclass'],axis=1,inplace=True)

titanic.head()

X=titanic.drop("Survived",axis=1)

y=titanic['Survived']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(X_train,y_train)

prediction = lr.predict(X_test)

from sklearn.metrics import classification_report

classification_report(y_test,prediction)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,prediction)

from sklearn.metrics import accuracy_score

accuracy_score(y_test,prediction)




#new logistic regression


from sklearn.datasets import make_classification

from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import pandas as pd

x,y=make_classification(
        n_samples=100,
        n_features=1,
        n_classes=2,
        n_clusters_per_class=1,
        flip_y=0.03,
        n_informative=1,
        n_redundant=0,
        n_repeated=0
        )

x
y

#plot

plt.scatter(x,y,c=y,cmap='rainbow')
plt.title('haha a scatterplot')
plt.show()


#split data

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=22)

X_train.shape

#logistic regression and prediction

lr=LogisticRegression()

lr.fit(X_train,y_train)

prediction = lr.predict(X_test)

lr.coef_
lr.intercept_

from sklearn.metrics import classification_report

classification_report(y_test,prediction)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,prediction)

from sklearn.metrics import accuracy_score

accuracy_score(y_test,prediction)