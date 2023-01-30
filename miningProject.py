# -*- coding: utf-8 -*-
"""
Created on Mon May 27 00:27:17 2019

@author: PAKISTAN
"""

#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd  

import numpy as np  

import matplotlib.pyplot as plt  

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:



dataset = pd.read_csv('D:/BS CS-7A/Data mining/train.csv')
test = pd.read_csv('D:/BS CS-7A/Data mining/test.csv')


# In[3]:


#dataset.head()
#test.head()


# In[4]:


#dataset.describe()


# # data Pre processing

# In[5]:


new=dataset.drop('PoolQC', axis=1)
new1=new.drop('Fence', axis=1)
new2=new1.drop('MiscFeature', axis=1)
new3=new2.drop('Alley', axis=1)
X_train = new3.drop('SalePrice', axis=1)  
new4=new3.drop('SalePrice', axis=1)
y_train = dataset['SalePrice']
t1=test.drop('PoolQC', axis=1)
t2=t1.drop('Fence', axis=1)
t3=t2.drop('MiscFeature', axis=1)
X_test=t3.drop('Alley', axis=1)
all_df = X_train.append(X_test, sort = False, ignore_index=True)
all_df


# # Check Missing Values

# In[6]:


a=all_df.isnull().sum()
a


# # handling missing values in Whole data

# In[7]:


all_df.fillna(all_df.mean(), inplace=True)
all_df["MSZoning"].fillna(0, inplace=True)
all_df["Utilities"].fillna(0, inplace=True)
all_df["Exterior1st"].fillna(0, inplace=True)
all_df["MasVnrType"].fillna(0, inplace=True)
all_df["BsmtQual"].fillna(0, inplace=True)
all_df["BsmtCond"].fillna(0, inplace=True)
all_df["BsmtExposure"].fillna(0, inplace=True)
all_df["Exterior2nd"].fillna(0, inplace=True)
all_df["BsmtFinType1"].fillna(0, inplace=True)
all_df["BsmtFinType2"].fillna(0, inplace=True)
all_df["KitchenQual"].fillna(0, inplace=True)
all_df["Functional"].fillna(0, inplace=True)
all_df["FireplaceQu"].fillna(0, inplace=True)
all_df["GarageType"].fillna(0, inplace=True)
all_df["GarageFinish"].fillna(0, inplace=True)
all_df["GarageQual"].fillna(0, inplace=True)
all_df["GarageCond"].fillna(0, inplace=True)
all_df["SaleType"].fillna(0, inplace=True)
all_df["Electrical"].fillna(0, inplace=True)

#X_test.columns[X_test.isnull().any()].tolist()  


# # Label Encoding in whole data

# In[8]:


all_df['MSZoning'],_ = pd.factorize(all_df['MSZoning'])
all_df['Street'],_ = pd.factorize(all_df['Street'])
all_df['LotShape'],_ = pd.factorize(all_df['LotShape'])
all_df['LandContour'],_ = pd.factorize(all_df['LandContour'])
all_df['Utilities'],_ = pd.factorize(all_df['Utilities'])
all_df['LotConfig'],_ = pd.factorize(all_df['LotConfig'])
all_df['LandSlope'],_ = pd.factorize(all_df['LandSlope'])
all_df['Neighborhood'],_ = pd.factorize(all_df['Neighborhood'])
all_df['Condition1'],_ = pd.factorize(all_df['Condition1'])
all_df['Condition2'],_ = pd.factorize(all_df['Condition2'])
all_df['BldgType'],_ = pd.factorize(all_df['BldgType'])
all_df['HouseStyle'],_ = pd.factorize(all_df['HouseStyle'])
all_df['RoofStyle'],_ = pd.factorize(all_df['RoofStyle'])
all_df['RoofMatl'],_ = pd.factorize(all_df['RoofMatl'])
all_df['Exterior1st'],_ = pd.factorize(all_df['Exterior1st'])
all_df['Exterior2nd'],_ = pd.factorize(all_df['Exterior2nd'])
all_df['MasVnrType'],_ = pd.factorize(all_df['MasVnrType'])
all_df['ExterQual'],_ = pd.factorize(all_df['ExterQual'])
all_df['ExterCond'],_ = pd.factorize(all_df['ExterCond'])
all_df['Foundation'],_ = pd.factorize(all_df['Foundation'])
all_df['BsmtQual'],_ = pd.factorize(all_df['BsmtQual'])
all_df['BsmtCond'],_ = pd.factorize(all_df['BsmtCond'])
all_df['BsmtExposure'],_ = pd.factorize(all_df['BsmtExposure'])
all_df['BsmtFinType1'],_ = pd.factorize(all_df['BsmtFinType1'])
all_df['BsmtFinType2'],_ = pd.factorize(all_df['BsmtFinType2'])
all_df['Heating'],_ = pd.factorize(all_df['Heating'])
all_df['HeatingQC'],_ = pd.factorize(all_df['HeatingQC'])
all_df['CentralAir'],_ = pd.factorize(all_df['CentralAir'])
all_df['Electrical'],_ = pd.factorize(all_df['Electrical'])
all_df['KitchenQual'],_ = pd.factorize(all_df['KitchenQual'])
all_df['Functional'],_ = pd.factorize(all_df['Functional'])
all_df['FireplaceQu'],_ = pd.factorize(all_df['FireplaceQu'])
all_df['GarageType'],_ = pd.factorize(all_df['GarageType'])
all_df['GarageFinish'],_ = pd.factorize(all_df['GarageFinish'])
all_df['GarageQual'],_ = pd.factorize(all_df['GarageQual'])
all_df['GarageCond'],_ = pd.factorize(all_df['GarageCond'])
all_df['PavedDrive'],_ = pd.factorize(all_df['PavedDrive'])
all_df['SaleType'],_ = pd.factorize(all_df['SaleType'])
all_df['SaleCondition'],_ = pd.factorize(all_df['SaleCondition'])
#all_df


# # Split in Training and testing after label incoding

# In[9]:


training_df=all_df[:1460]
testing_df=all_df[-1459:]


# # Linear Regression

# In[10]:


#all_df.info()


# In[11]:


from sklearn.linear_model import LinearRegression


# In[12]:


reg=LinearRegression()
reg.fit(training_df, y_train)


# In[13]:


y_pred=reg.predict(testing_df)


# # Training accuracy with Linear Regression

# In[14]:


y_pred1=reg.predict(training_df)
from sklearn import metrics as ms
accuracy = ms.r2_score(y_train,y_pred1)
print('training accuracy with linear regression ',accuracy*100,'%')
df1=pd.DataFrame({'SalePrice':y_pred})
#df1.to_csv(r'D:/BS CS-7A/Data mining/submit2.csv')


# # Random Forest

# In[21]:


from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics as ms
from pprint import pprint
from sklearn.model_selection import GridSearchCV
regressor = RandomForestRegressor(n_estimators=20, random_state=0)  
#param_grid = {'bootstrap': [True],
#    'max_depth': [50,60,70,80, 90],
#    'max_features': [3, 4,5,6],
#    'min_samples_leaf': [1,2, 3, 4],
#    'min_samples_split': [6,8, 10, 12],
#    'n_estimators': [50,100, 200, 300, 500]}
param_grid={'bootstrap': [True],
 'max_depth': [50],
 'max_features': [6],
 'min_samples_leaf': [1],
 'min_samples_split': [6],
 'n_estimators': [500]}
grid_search = GridSearchCV(estimator = regressor, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
regressor.fit(training_df,y_train)
y_pred1=regressor.predict(training_df)
grid_search.fit(training_df, y_train)
a_pred=grid_search.predict(testing_df)
#best_grid = grid_search.best_estimator_
grid_search.best_params_
grid_search.cv_results_
#df1=pd.DataFrame({'SalePrice predicted with random forest ':a_pred})
#df1.to_csv(r'D:/BS CS-7A/Data mining/submitnew.csv')


# # Training accuracy with Random forest

# In[16]:


accuracy = ms.r2_score(y_train,y_pred1)
print('Training accuracy ',accuracy*100,'%')
best_result = grid_search.best_score_ 
print('Test Accuracy with random forest',best_result*100,'%')


# # Output

# In[17]:


df=pd.DataFrame({'SalePrice predicted with Linear regression ':y_pred,'SalePrice predicted with random forest ':a_pred})
df


# In[18]:


f, ax = plt.subplots(figsize=(18,10)) # set the size that you'd like (width, height)
plt.bar(testing_df['Id'],a_pred)


# In[19]:


f, ax = plt.subplots(figsize=(18,10)) # set the size that you'd like (width, height)
plt.bar(testing_df['Id'],y_pred)


# In[ ]:





# In[ ]:




