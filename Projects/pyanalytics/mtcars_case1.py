#Case Study on mtcars dataset in Python	download data

#Download data
import statsmodels.api as sm
#https://vincentarelbundock.github.io/Rdatasets/datasets.html
dataset_mtcars = sm.datasets.get_rdataset(dataname='mtcars', package='datasets')
dataset_mtcars.data.head()
mtcars = dataset_mtcars.data
mtcars
#structure

mtcars.shape

#summary

mtcars.describe()

#print first / last few rows

mtcars.head()
mtcars.tail()


#print number of rows

mtcars['mpg'].count()

#number of columns

len(mtcars.columns)

#print names of columns

mtcars.columns

#Filter Rows

#cars with cyl=8

mtcars[mtcars['cyl']==8]

#cars with mpg <= 27

mtcars[mtcars['mpg']<=27]

#rows match auto tx

mtcars[mtcars['am']==1]

#First Row

mtcars.head(1)

#last Row

mtcars.tail(1)

# 1st, 4th, 7th, 25th row + 1st 6th 7th columns.

mtcars.iloc[0,[1,6,7]]
mtcars.iloc[3,[1,6,7]]
mtcars.iloc[6,[1,6,7]]
mtcars.iloc[24,[1,6,7]]

# first 5 rows and 5th, 6th, 7th columns of data frame

mtcars.iloc[0:5,[5,6,7]]

#rows between 25 and 3rd last

mtcars.iloc[24:len(mtcars)-4,:]

#alternative rows and alternative column    
    
for i in xrange(0,len(mtcars),2):
    print(mtcars.iloc[i,[0,2,4,6,8,10]])

#find row with Mazda RX4 Wag and columns cyl, am

mtcars.loc['Mazda RX4 Wag']

#find row betwee Merc 280 and Volvo 142E Mazda RX4 Wag and columns cyl, am

mtcars.iloc[10:31,2:8]

# mpg > 23 or wt < 2
#with or condition
c1=mtcars['mpg'] > 23.0
c2=mtcars['wt']<2.0
c1

mtcars[c1 | c2]

mtcars[c1]
mtcars[c2]
#using lambda for above

x = lambda a, b : mtcars[a | b]
print(x(c1,c2))

#find unique rows of cyl, am, gear

mtcars.cyl.unique()
mtcars.am.unique()
mtcars.gear.unique()

#create new columns: first make a copy of mtcars to mtcars2

mtcars2=mtcars

mtcars2.shape

#keeps other cols and divide displacement by 61

mtcars2['disp']=mtcars2['disp']/61
mtcars2['disp']

# multiple mpg * 1.5 and save as original column

mtcars2['mpg']=mtcars2['mpg']*1.5
mtcars2['mpg']

