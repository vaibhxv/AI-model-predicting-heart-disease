#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib as plt
import numpy as np
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical


# In[5]:


df = pd.read_csv("E:/heart.csv")


# In[6]:


df


# In[7]:


X = df.drop(["target"], axis=1)


# In[8]:


y= df["target"]
target = to_categorical(y)


# In[18]:


model = Sequential()
model.add(Dense(840, activation = "relu", input_shape = (13, )))
model.add(Dense(820))
model.add(Dense(500))
model.add(Dense(2, activation = "softmax"))


# In[19]:


model.compile(optimizer="adam", loss= "categorical_crossentropy", metrics= ["accuracy"])


# In[29]:


model.fit(X, target, epochs=20 )


# In[30]:



X


# In[31]:


data= pd.DataFrame({'age':[21], 'sex':[0], 'cp':[2], 'trestbps':[140], 'chol':[140], 'fbs':[1], 'restecg':[0], 'thalach':[110], 'exang':[0], 'oldpeak':[0], 'slope':[1], 'ca':[1], 'thal':[2]})


# In[32]:


data


# In[33]:


model.predict(data)


# In[ ]:




