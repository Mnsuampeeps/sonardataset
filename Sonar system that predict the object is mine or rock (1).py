#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from scipy import stats
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[3]:


# data collection & data processing 
# load dataset using pandas dataframe
sonar_data = pd.read_csv('sonar.csv', header=None)


# In[4]:


# show data using pandas 
sonar_data.head()


# In[5]:


# number of rows and columns 
sonar_data.shape


# In[6]:


# describe--> statistical measure of the data 
sonar_data.describe()


# In[7]:


sonar_data[60].value_counts()


# In[8]:


sonar_data[60].value_counts().plot(kind='bar')


# In[9]:


sonar_data.groupby(60).mean()
# m--> Mine
# R--> rock


# In[10]:


# separating data and labels 
# it is supervised learning to use labels data
x = sonar_data.drop(columns=60, axis=1)
y = sonar_data[60]


# In[11]:


print(x)
print(y)


# In[12]:


# training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, stratify = y, random_state=1)


# In[13]:


print(x.shape, x_train.shape, x_test.shape)


# In[14]:


print(x_train)
print(y_train)


# In[ ]:





# In[15]:


#model traning ==> logistic regression
model = LogisticRegression()


# In[16]:


# Train the model on the training data
model.fit(x_train, y_train)


# In[17]:


#accuracy on training data 
x_train_prediction = model.predict(x_train)
x_train_accuracy = accuracy_score(x_train_prediction, y_train)


# In[18]:


print('accuracy on training data: ', x_train_accuracy)


# In[19]:


#accuracy on test data 
x_test_prediction = model.predict(x_test)
x_test_accuracy = accuracy_score(x_test_prediction, y_test)


# In[20]:


print('accuracy on test data: ', x_test_accuracy)


# In[21]:




input_data = [0.0115,0.0150,0.0136,0.0076,0.0211,0.1058,0.1023,0.0440,0.0931,0.0734,0.0740,0.0622,0.1055,0.1183,0.1721,0.2584,0.3232,0.3817,0.4243,0.4217,0.4449,0.4075,0.3306,0.4012,0.4466,0.5218,0.7552,0.9503,1.0000,0.9084,0.8283,0.7571,0.7262,0.6152,0.5680,0.5757,0.5324,0.3672,0.1669,0.0866,0.0646,0.1891,0.2683,0.2887,0.2341,0.1668,0.1015,0.1195,0.0704,0.0167,0.0107,0.0091,0.0016,0.0084,0.0064,0.0026,0.0029,0.0037,0.0070,0.0041]

# Convert input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape data
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# Make a prediction using your model
prediction = model.predict(input_data_reshaped)

# Print the prediction
print(prediction)
if (prediction[0]== "R"):
  print('The object is a ROCK')
else: 
    print('The Object is MINE')


# In[22]:


input_data = [0.0180,0.0444,0.0476,0.0698,0.1615,0.0887,0.0596,0.1071,0.3175,0.2918,0.3273,0.3035,0.3033,0.2587,0.1682,0.1308,0.2803,0.4519,0.6641,0.7683,0.6960,0.4393,0.2432,0.2886,0.4974,0.8172,1.0000,0.9238,0.8519,0.7722,0.5772,0.5190,0.6824,0.6220,0.5054,0.3578,0.3809,0.3813,0.3359,0.2771,0.3648,0.3834,0.3453,0.2096,0.1031,0.0798,0.0701,0.0526,0.0241,0.0117,0.0122,0.0122,0.0114,0.0098,0.0027,0.0025,0.0026,0.0050,0.0073,0.0022]

# Convert input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape data
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# Make a prediction using your model
prediction = model.predict(input_data_reshaped)

# Print the prediction
print(prediction)
if (prediction[0]== "M"):
  print('The object is a MINE')
else: 
    print('The Object is ROCK')


# In[33]:


# Define a KNN classifier with 5 neighbors
classifier= KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)  
classifier.fit(x_train, y_train)

#Predicting the result  
x_pred_train= classifier.predict(x_train) 
print("KNN model accuracy of training data is (in %):", metrics.accuracy_score(y_train, x_pred_train)*100)



# In[32]:


x_pred_test= classifier.predict(x_test) 
print("KNN model accuracy of test data is (in %):", metrics.accuracy_score(y_test, x_pred_test)*100)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




