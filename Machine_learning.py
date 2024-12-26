#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_california_housing


# In[2]:


dataset=fetch_california_housing()
print(dataset)


# In[15]:


print(dataset.feature_names)


# In[23]:


dataset.target


# In[24]:


import pandas as pd


# In[37]:


df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df.head()
df['Price']=dataset.target
df.head()


# In[31]:


df.isnull().sum()


# In[39]:


df.head()


# In[40]:


#independent and dependent
X=df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[83]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train,y_train)
# regression.coef_
# regression.intercept_
y_pred = regression.predict(X_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy as np

mse = mean_squared_error(y_test,y_pred)
print("mse :",mse)
mbe = mean_absolute_error(y_test,y_pred)
print("mbe :", mbe)
rmse=np.sqrt(mse)
print("rmse:", rmse)

## Accuracy r2 and adjusted r square
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_pred)
print("r2  :", r2)
#display adjusted R-squared
adr2=1 - (1-r2)*(len(y)-1)/(len(y)-X.shape[1]-1)
print("adr2:", adr2)


# In[90]:


from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(X_train,y_train)
y_pred_rid = ridge.predict(X_test)


mse_rid=mean_squared_error(y_test,y_pred_rid)
print(mse_rid)
mae_rid=mean_absolute_error(y_test,y_pred_rid)
print(mae_rid)
print(np.sqrt(mse_rid))


# In[94]:


from sklearn.linear_model import Lasso
lasso = Lasso()
lasso = lasso.fit(X_train,y_train)
y_pred_las =lasso.predict(X_test)

mse_las=mean_squared_error(y_test,y_pred_las)
print(mse_las)
mae_las=mean_absolute_error(y_test,y_pred_las)
print(mae_las)
print(np.sqrt(mse_las))


# In[ ]:




