#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras 
from sklearn.metrics import  accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


def wrangle(path):
    #import data
    df = pd.read_csv(path)
    
    # get day, hour and month
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], dayfirst=True)
    df['trans_day'] = df['trans_date_trans_time'].dt.weekday
    df['trans_hour'] = df['trans_date_trans_time'].dt.hour
    df['trans_month'] = df['trans_date_trans_time'].dt.month
    
    # encoding categorical  features
    df['category'] = pd.Categorical(df.category).codes
    df['gender'] = pd.Categorical(df.gender).codes
    df['state'] =  pd.Categorical(df.state).codes
    
    # finding age 
    df['dob'] = pd.to_datetime(df['dob'], dayfirst =True)
    df['age'] =  df['trans_date_trans_time'].dt.year - df['dob'].dt.year
    
    
    #dropping features  not needed or with high cardinality
    high_cardinality = ['cc_num','lat','trans_num', 'unix_time', 'long','zip','city_pop','job', 'merchant', 'first', 'last','street','city', 'trans_num', 'merch_lat', 'merch_long']
    not_needed = ['trans_date_trans_time','dob']
    
    df.drop(columns = high_cardinality +not_needed, inplace = True)
    
    return  df


# In[3]:


df = wrangle('fraudTrain.csv')
df.head(3)


# In[4]:


df.shape


# In[5]:


df.dtypes


# In[6]:


sns.heatmap(df.corr(), cmap = 'coolwarm', annot = True)


# In[7]:


X_train = df.drop(columns=['is_fraud'])
y_train = df['is_fraud']


# In[8]:



print('is fraud', y_train.sum())
# percentage of fraud in data
print('percentage of fraud in data', y_train.sum()/ len(y_train)) 


# In[9]:


model = keras.Sequential([
    keras.layers.Dense(20, input_shape = (9,), activation = 'relu'),
    keras.layers.Dense(15, activation = 'relu'),
    keras.layers.Dense(1,  activation = 'sigmoid')
])

model.compile(optimizer = 'adam' ,loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 10, batch_size = 512)


# In[10]:


df_test = wrangle('fraudTest.csv')
X_test = df_test.drop(columns=['is_fraud'])
y_test = df_test['is_fraud']


# In[11]:


df_test.head(5)


# In[12]:


print('X_test shape:', X_test.shape)
print( 'y_test shape:', y_test.shape)


# In[13]:


model.summary()


# In[14]:


model.evaluate(X_test, y_test)


# In[15]:


y_pred = pd.Series(model.predict(X_test).flatten())
y_pred= (y_pred>0.5).astype(int)
y_pred[:5]


# In[16]:


accuracy_score(y_test, y_pred)


# In[17]:


cm = tf.math.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True)


# In[18]:


cm


# In[19]:


clf = classification_report(y_test, y_pred, output_dict = True)
print(clf)
sns.heatmap(pd.DataFrame(clf), annot= True)


# In[ ]:





# In[ ]:




