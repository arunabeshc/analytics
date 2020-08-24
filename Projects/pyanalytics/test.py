import numpy as np
import statsmodels.api as sm

from statsmodels.tools import add_constant
x = [[0,1], [5,1], [15,2], [25,2], [35,11], [45,15], [55,34], [60,35]]
x
y = [4,5,20,14,32,22,38,43]
y

x= sm.add_constant(x)  #constant term of 1 added
x
model3 = sm.OLS(y,x)
model3
results = model3.fit()
results
results.summary()
results.rsquared  #coeff of determination
results.rsquared_adj 
results.params  #bo, b1, b2

results.fittedvalues
results.predict(x)



#!pip install RegscorePy
import RegscorePy
#aic(y, y_pred, p)
RegscorePy.aic.aic(y=y, y_pred= results.predict(x), p=1)
RegscorePy.bic.bic(y=y, y_pred= results.predict(x), p=1)
