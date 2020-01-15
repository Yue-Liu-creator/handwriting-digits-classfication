#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


from sklearn.datasets import load_digits


# In[15]:


data1=load_digits()


# In[67]:


data1


# In[16]:


data=pd.DataFrame(data1["data"])


# In[17]:


target = pd.Series(data1["target"])


# In[21]:


target


# In[22]:


row=[1,100,200,300,1000,1100,1200,1300]


# In[24]:


image=[]
for i in row:
    image.append(data.iloc[i].values.reshape(8,8))


# In[25]:


image


# In[28]:


fix,axs= plt.subplots(2,4)
axs[0,0].imshow(image[0])
axs[0,1].imshow(image[1])
axs[0,2].imshow(image[2])
axs[0,3].imshow(image[3])
axs[1,0].imshow(image[4])
axs[1,1].imshow(image[5])
axs[1,2].imshow(image[6])
axs[1,3].imshow(image[7])


# In[46]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


# In[47]:


def train(features, target, k):
    model=KNeighborsClassifier(n_neighbors=k)
    model.fit(features,target)
    return model


# In[48]:


def test(model,features,target):
    prediction = model.predict(features)
    score=accuracy_score(target,prediction)
    return score


# In[51]:


def cross_validate(features, target,k ):
    accuracy=[]
    kk=KFold(n_splits=4)
    for train_index, test_index in kk.split(features.index):
        model=train(features.iloc[train_index],target[train_index],k)
        score=test(model,features.iloc[test_index],target[test_index])
        accuracy.append(score)
    avg_score=np.mean(accuracy)
    return avg_score


# In[64]:


y=[]
for k in range(2,20):
    result = cross_validate(data,target,k)
    y.append(result)


# In[65]:


k=range(2,20)
k


# In[66]:


plt.plot(k,y)


# In[68]:


from sklearn.neural_network import MLPClassifier


# In[69]:


def nn_train(features, target,k):
    model=MLPClassifier(hidden_layer_sizes=(k,))
    model.fit(features,target)
    return model


# In[70]:


def nn_test(model,features,target):
    prediction = model.predict(features)
    score=accuracy_score(target,prediction)
    return score


# In[71]:


def nn_cross_validate(features, target,k ):
    accuracy=[]
    kk=KFold(n_splits=4)
    for train_index, test_index in kk.split(features.index):
        model=nn_train(features.iloc[train_index],target[train_index],k)
        score=nn_test(model,features.iloc[test_index],target[test_index])
        accuracy.append(score)
    avg_score=np.mean(accuracy)
    return avg_score


# In[72]:


neurons=[8,16,32,64,128,56]


# In[73]:


accuracy=[]
for n in neurons:
    result= nn_cross_validate(data,target,n)
    accuracy.append(result)


# In[75]:


plt.plot(range(1,7),accuracy)


# In[76]:


# model with 128 neurons starts to overfit as the accuracy begins to decrease after arriving the submit.


# In[78]:


np.max(accuracy) 


# In[79]:


np.max(y)


# In[80]:


# the highest accuracy for k nearest neighbours is higher than that for neural network model. so k nearest beighbors model is better


# ### double layer 

# In[84]:


def nn_two_train(features, target,k):
    model=MLPClassifier(hidden_layer_sizes=(k,k))
    model.fit(features,target)
    return model


# In[85]:


def nn_two_test(model,features,target):
    prediction = model.predict(features)
    score=accuracy_score(target,prediction)
    return score


# In[86]:


def nn_two_cross_validate(features, target,k ):
    accuracy=[]
    kk=KFold(n_splits=4)
    for train_index, test_index in kk.split(features.index):
        model=nn_train(features.iloc[train_index],target[train_index],k)
        score=nn_test(model,features.iloc[test_index],target[test_index])
        accuracy.append(score)
    avg_score=np.mean(accuracy)
    return avg_score


# In[87]:


result = nn_two_cross_validate(data,target,64)


# In[88]:


result


# In[89]:


# when the single layer with 64 node, we get the best result for neural network model, but when we choose double layer all with 64 nodes, the accuracy starts to decrease. this means the model that has two layers with both 64nodes start to overfit


# ### three layer 

# In[91]:


def nn_three_train(features, target,k):
    model=MLPClassifier(hidden_layer_sizes=(k,k,k))
    model.fit(features,target)
    return model


# In[92]:


def nn_three_test(model,features,target):
    prediction = model.predict(features)
    score=accuracy_score(target,prediction)
    return score


# In[109]:


def nn_three_cross_validate(features, target,k ):
    accuracy=[]
    kk=KFold(n_splits=4)
    for train_index, test_index in kk.split(features.index):
        model=nn_train(features.iloc[train_index],target[train_index],k)
        score=nn_test(model,features.iloc[test_index],target[test_index])
        accuracy.append(score)
    avg_score=np.mean(accuracy)
    return avg_score


# In[110]:


accuracy=[]
for i in [10,64,128]:
    result= nn_three_cross_validate(data,target,i)
    accuracy.append(accuracy)  
accuracy


# ### decision tree

# In[106]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[132]:


def dt_train(features, target,k):
    model=DecisionTreeClassifier(random_state=1,max_depth=k)
    model.fit(features,target)
    return model


# In[133]:


def dt_test(model,features,target):
    prediction = model.predict(features)
    score=accuracy_score(target,prediction)
    return score


# In[134]:


def df_cross_validate(features, target,k):
    accuracy=[]
    kk=KFold(n_splits=4)
    for train_index, test_index in kk.split(features.index):
        model=dt_train(features.iloc[train_index],target[train_index],k)
        score=dt_test(model,features.iloc[test_index],target[test_index])
        accuracy.append(score)
    avg_score=np.mean(accuracy)
    return avg_score


# In[136]:


accuracy =[]
for k in range(1,20):
    result = df_cross_validate(data,target,k)
    accuracy.append(result)
accuracy


# In[137]:


plt.plot(range(1,20),accuracy)


# ### random forest

# In[139]:


def rf_train(features, target,k):
    model=RandomForestClassifier(random_state=1,max_depth=k)
    model.fit(features,target)
    return model


# In[141]:


def rf_test(model,features,target):
    prediction = model.predict(features)
    score=accuracy_score(target,prediction)
    return score


# In[144]:


def rf_cross_validate(features, target,k):
    accuracy=[]
    kk=KFold(n_splits=4)
    for train_index, test_index in kk.split(features.index):
        model=rf_train(features.iloc[train_index],target[train_index],k)
        score=rf_test(model,features.iloc[test_index],target[test_index])
        accuracy.append(score)
    avg_score=np.mean(accuracy)
    return avg_score


# In[145]:


accuracy =[]
for i in range(1,20):
    result= rf_cross_validate(data,target,i)
    accuracy.append(result)
accuracy


# In[146]:


plt.plot(range(1,20),accuracy)


# In[ ]:




