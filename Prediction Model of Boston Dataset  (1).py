
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# In[2]:


boston= load_boston()


# In[3]:


dir(boston)


# In[4]:


df=pd.DataFrame(boston.data, columns= boston.feature_names)
df.head()


# In[5]:


boston.target


# In[6]:


df['target']= boston.target
df.head()


# In[7]:


x = df.drop('target',axis=1)
x.head()


# In[8]:


y = df['target']
y.head()


# In[9]:


x_train, x_test,y_train,y_test= train_test_split(x,y,test_size=0.2)


# In[10]:


x_train.shape


# In[11]:


x_test.shape


# In[12]:


model = GradientBoostingRegressor(n_estimators=2, learning_rate=1)


# In[13]:


model.fit(x_train,y_train)


# In[14]:


model.score(x_test,y_test)


# In[15]:


y_pred = model.predict(x_test)
y_pred


# In[16]:


r2_score(y_test,y_pred)


# In[17]:


from sklearn.model_selection import GridSearchCV


# In[18]:


LR = {'learning_rate':[0.15,0.1,0.10,0.05], 'n_estimators':[100,150,200,250]}
tuning = GridSearchCV(estimator= GradientBoostingRegressor(), param_grid=LR, scoring='r2')
tuning.fit(x_train,y_train)


# In[19]:


tuning.best_params_


# In[36]:


tuning.best_score_

