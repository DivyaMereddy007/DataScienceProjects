
# coding: utf-8

# In[42]:


from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import train_test_split
import pydotplus
import pandas as pd
from sklearn.externals.six import StringIO   
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pydotplus
import pandas as pd
from sklearn.externals.six import StringIO   
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier


# In[43]:


#read CSV files of 2 data sets
df = pd.read_csv("C:/Users/mereddda/Desktop/IDA/breast-cancer-wisconsin.csv", skipinitialspace=True)    
df.drop(df.columns[[1]], axis=1, inplace=True)
df.head()


# In[44]:


X=df.iloc[:,2:]
X.head()


# In[45]:


Y=df.iloc[:,1]
Y.head()


# In[75]:


import matplotlib.pyplot as plt
import numpy as np


# In[76]:


my_data = np.genfromtxt('C:/Users/mereddda/Desktop/IDA/breast-cancer-wisconsin.csv', delimiter=',') # read the data
X = my_data[:, 0].reshape(-1,1) # -1 tells numpy to figure out the dimension by itself
ones = np.ones([X.shape[0], 1]) # create a array containing only ones 
X = np.concatenate([ones, X],1) # cocatenate the ones to X matrix
y = my_data[:, 1].reshape(-1,1) # create the y matrix


# In[77]:


my_data


# In[61]:


ones


# In[62]:


X


# In[63]:


y = my_data[:, 1].reshape(-1,1)
y


# In[56]:


plt.scatter(my_data[:, 0].reshape(-1,1), y)


# In[67]:


alpha = 0.0001
iters = 1000

# theta is a row vector
theta = np.array([[1.0, 1.0]])


# In[69]:


def computeCost(X, y, theta):
    inner = np.power(((X @ theta.T) - y), 2) # @ means matrix multiplication of arrays. If we want to use * for multiplication we will have to convert all arrays to matrices
    return np.sum(inner) / (2 * len(X))


# In[70]:


computeCost(X, y, theta) 


# In[71]:



def gradientDescent(X, y, theta, alpha, iters):
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum((X @ theta.T - y) * X, axis=0)
        cost = computeCost(X, y, theta)
        # if i % 10 == 0: # just look at cost every ten loops for debugging
        #     print(cost)
    return (theta, cost)


# In[73]:


g, cost = gradientDescent(X, y, theta, alpha, iters)  
print(g, cost)


# In[74]:


plt.scatter(my_data[:, 0].reshape(-1,1), y)
axes = plt.gca()
x_vals = np.array(axes.get_xlim()) 
y_vals = g[0][0] + g[0][1]* x_vals #the line equation
plt.plot(x_vals, y_vals, '--')

