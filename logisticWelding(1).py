#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.metrics import r2_score


from sklearn.model_selection import train_test_split


# In[3]:


data=pd.read_csv('weld_Quality.csv')
data.head()


# In[5]:


data.info()


# In[4]:


data = data.rename(columns = {"Dataset": "Weld_Quality"}) 


# In[7]:



df=data.replace([1, 2 ,3], ['Lack of Fusion', 'Good Weld', 'Burn Through'])
sns.countplot(x=df['Weld_Quality'])


# In[8]:


sns.countplot(y=df['Volts'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[5]:


x=data.drop(['Weld_Quality','SL. NO'],axis='columns')


# In[6]:


y = data['Weld_Quality']


# In[7]:


y.head()


# In[8]:


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=.25,random_state=7)


# In[9]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x,y)


# In[10]:


model.score(x,y)


# In[11]:


res=model.predict([[100,20,3,1]])


# In[12]:


res


# In[ ]:




