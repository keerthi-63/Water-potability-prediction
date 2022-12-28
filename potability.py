#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
warnings.filterwarnings('ignore')
sns.set()


# In[4]:


df=pd.read_csv(r'D:\Vit\Datasets\water_potability.csv')
df.head()


# In[5]:


def detect_outliers_iqr(data):
    outliers = []
    data = sorted(data)
    q1 = np.quantile(data, 0.25)
    q3 = np.quantile(data, 0.75)
    #print(q1, q3)
    IQR = q3-q1
    lwr_bound = q1-(1.5*IQR)
    upr_bound = q3+(1.5*IQR)
    print(lwr_bound, upr_bound)
    for i in data: 
        if (i<lwr_bound or i>upr_bound):
            outliers.append(i)
            #print(i)
    return outliers


# In[6]:


a=['ph','Sulfate','Trihalomethanes']
df[a] = df[a].fillna(df[a].median())
df.head()


# In[7]:


columns = ['ph','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity']
for i in columns: 
  print("Outliers in", i,":", len(detect_outliers_iqr(df[i])))


# In[8]:


df_new = df.copy()
for i in columns:
  Q1 = df[i].quantile(0.25)
  Q3 = df[i].quantile(0.75)
  IQR=Q3-Q1
  upper_limit = Q3 + 1.5 * IQR
  lower_limit = Q1 - 1.5 * IQR
  df_new[i] = np.where(df_new[i] > upper_limit,upper_limit,np.where(df_new[i] < lower_limit,lower_limit,df_new[i]))


# In[9]:


for i in columns: 
  print("Outliers in", i,":", len(detect_outliers_iqr(df_new[i])))


# In[10]:


df.head()


# In[16]:


water_potability = {
    'No': 0,
    'Yes': 1
}
df1 = df.replace({'Potability': water_potability})
df1.head()


# In[21]:


X = df1.iloc[:, :9]
Y = df1.iloc[:, 9]


# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=22)


# In[24]:


from sklearn.ensemble import RandomForestClassifier
reg_rf = RandomForestClassifier()
reg_rf.fit(X_train, Y_train)


# In[25]:


y_pred = reg_rf.predict(X_test)


# In[26]:


reg_rf.score(X_train, Y_train)


# In[27]:



# In[32]:


pickle_out = open("reg_rf.pkl", "wb") 
pickle.dump(reg_rf, pickle_out) 
pickle_out.close()


# In[33]:


pickle_in = open('reg_rf.pkl', 'rb')
classifier = pickle.load(pickle_in)


# In[34]:


#st.sidebar.header(' Check water potability')
#select = st.sidebar.selectbox('Select Form', ['Form 1'], key='select')
#if not st.sidebar.checkbox("Hide", True, key='sidebar_1'):
st.title('Water potability prediction')
ph = st.slider("Enter pH value:",0.0,14.0,key='ph')
Hardness = st.slider("Enter Hardness value:",0.0,500.0,key='Hardness')
Solids =  st.slider("Enter Solids value:",0.0,90000.0,key='Solids')
Chloramines = st.slider("Enter Chloramines value:",0.0,20.0,key='Chloramines')
Sulfate = st.slider("Enter Sulfate value:",0.0,500.0,key='Sulfate')
Conductivity = st.slider("Enter Conductivity value:",0.0,1000.0,key='Conductivity')
Organic_carbon = st.slider("Enter Organic Carbon value:",0.0,50.0,key='Organic_carbon')
Trihalomethanes = st.slider("Enter Trihalomethanes value:",0.0,200.0,key='Trihalomethanes')
Turbidity = st.slider("Enter Turbidity value:",0.0,20.0,key='Turbidity')
prediction = classifier.predict([[ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]])

if st.button('Predict'):
        
       
    if prediction == 0:
        st.error('The water is not potable')
     
    else:
        st.success('The water is potable')
     
    #st.write("Accuracy "+str(reg_rf.score(X_test, Y_test)))


# In[ ]:




