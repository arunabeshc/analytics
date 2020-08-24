#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 02:53:22 2020

@author: riju
"""

# Linear Regression -1 Marketing Data - Sales - YT, FB, print
#libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model #1st method
import statsmodels.api as sm  #2nd method
import matplotlib.pyplot as plt
import seaborn as sns

url ='https://raw.githubusercontent.com/DUanalytics/datasets/master/R/marketing.csv'
marketing = pd.read_csv(url)
marketing.head()

#describe data

marketing.columns
marketing.describe
marketing.shape
marketing.info()
marketing.isnull().sum()  #each col
sns.heatmap(marketing.isnull(),yticklabels='False')

#visualise few plots to check correlation

sns.scatterplot(x="youtube",y="sales",data=marketing)

sns.scatterplot(x="facebook",y="sales",data=marketing)

sns.scatterplot(x="newspaper",y="sales",data=marketing)

sns.barplot(data = marketing, x='youtube', y='sales')

#split data into train and test

X= marketing[['youtube', 'facebook' , 'newspaper']]
X
y = marketing['sales']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

#build the model

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X,y)
lm.score(X,y)
lm.coef_
lm.intercept_


#predict on test values

prediction = lm.predict(X_test)

prediction

plt.scatter(prediction,y_test,cmap='rainbow')
plt.title('a scatterplot')
plt.show()

#find metrics - R2, Adjt R2, RMSE, MAPE etc

import numpy as np
import statsmodels.api as sm

from statsmodels.tools import add_constant

X = sm.add_constant(X)
lm_1 = sm.OLS(y, X).fit()
print(lm_1.summary())

error = np.sum(np.abs(np.subtract(y_test,prediction)))
sum1 = np.sum(prediction)
MAPE = error/sum1
print("mean absolute percentage error = ",MAPE)


#predict on new value

newdata = pd.DataFrame({'youtube':[50,60,70]})
newdata['facebook']=[20,30,40]
newdata['newspaper']=[70,75,80]
newdata

prediction2 = lm.predict(newdata)
prediction2

#your ans should be close to [ 9.51, 11.85, 14.18] 

#conclude by few lines

newdata['predicted_sales']=prediction2

newdata