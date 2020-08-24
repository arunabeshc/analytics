#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 00:42:07 2020

@author: riju
"""

#import pandas

import pandas as pd

#load the csv file

data = pd.read_csv('/Users/riju/Desktop/IIM_Calcutta_Study/IIMC_Study/EXTRAS !/CPBA/analytics/Projects/pyanalytics/denco.csv')

data.head()

data.tail(10)

#most loyal customers

data['custname'].value_counts().head(10)

#Which customers contribute the most to their revenue

data.groupby('custname')['revenue'].sum().sort_values(ascending=False).head(1)

#What part numbers bring in to significant portion of revenue

data.groupby('partnum')['revenue'].sum().sort_values(ascending=False).head()

#What parts have the highest profit margin ?

data.groupby('partnum')['margin'].sum().sort_values(ascending=False).head()

#Who are their top buying customers?

data.groupby('custname')['revenue'].sum().sort_values(ascending=False).head()

#Who are the customers who are bringing more revenue?

data.groupby('custname')['margin'].sum().sort_values(ascending=False).head()