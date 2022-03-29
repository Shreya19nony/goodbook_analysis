#!/usr/bin/env python
# coding: utf-8

# In[27]:


get_ipython().system('pip install textblob')


# In[37]:


import pandas as pd
import numpy as np
from textblob import TextBlob


# In[52]:


data=pd.read_csv(r'C:\Users\SHREYA\Downloads\Title_sentiment2.csv')


# In[53]:


data.head()


# In[57]:


data.dtypes


# In[61]:


data = data.astype(str)


# In[60]:


data.dtypes


# In[65]:


data['TextBlob_Subjectivity']=data['original_title'].apply(lambda x:TextBlob(x).sentiment.subjectivity)
data['TextBlob_Polarity']=data['original_title'].apply(lambda x:TextBlob(x).sentiment.polarity)


# In[67]:


data['TextBlob_Analysis']=data['TextBlob_Polarity'].apply(lambda x:'negative' if x<0 else 'positive')


# In[71]:


data.head()


# In[74]:


data['TextBlob_Analysis'].value_counts()

