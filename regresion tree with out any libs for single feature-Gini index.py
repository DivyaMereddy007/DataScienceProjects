
# coding: utf-8

# In[1]:


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


# In[4]:


#read CSV files of 2 data sets
df = pd.read_csv("C:/Users/mereddda/Desktop/IDA/breast-cancer-wisconsin.csv", skipinitialspace=True)    
df.drop(df.columns[[1]], axis=1, inplace=True)
df.head()


# In[5]:


X=df.iloc[:,2:]
X.head()


# In[6]:


Y=df.iloc[:,1]
Y.head()


# In[7]:


import pandas as pd
from math import pow


# In[8]:


def get_headers(dataframe):
    """
    Get the headers name of the dataframe
    :param dataframe:
    :return:
    """
    return dataframe.columns.values


# In[9]:


get_headers(df)


# In[10]:


def cal_mean(readings):
    """
    Function to calculate the mean value of the input readings
    :param readings:
    :return:
    """
    readings_total = sum(readings)
    number_of_readings = len(readings)
    mean = readings_total / float(number_of_readings)
    return mean


# In[11]:


def cal_variance(readings):
    """
    Calculating the variance of the readings
    :param readings:
    :return:
    """
 
    # To calculate the variance we need the mean value
    # Calculating the mean value from the cal_mean function
    readings_mean = cal_mean(readings)
    # mean difference squared readings
    mean_difference_squared_readings = [pow((reading - readings_mean), 2) for reading in readings]
    variance = sum(mean_difference_squared_readings)
    return variance / float(len(readings) - 1)


# In[12]:


def cal_covariance(readings_1, readings_2):
    """
    Calculate the covariance between two different list of readings
    :param readings_1:
    :param readings_2:
    :return:
    """
    readings_1_mean = cal_mean(readings_1)
    readings_2_mean = cal_mean(readings_2)
    readings_size = len(readings_1)
    covariance = 0.0
    for i in xrange(0, readings_size):
        covariance += (readings_1[i] - readings_1_mean) * (readings_2[i] - readings_2_mean)
    return covariance / float(readings_size - 1)
 


# In[13]:


def cal_simple_linear_regression_coefficients(x_readings, y_readings):
    """
    Calculating the simple linear regression coefficients (B0, B1)
    :param x_readings:
    :param y_readings:
    :return:
    """
    # Coefficient B1 = covariance of x_readings and y_readings divided by variance of x_readings
    # Directly calling the implemented covariance and the variance functions
    # To calculate the coefficient B1
    b1 = cal_covariance(x_readings, y_readings) / float(cal_variance(x_readings))
 
    # Coefficient B0 = mean of y_readings - ( B1 * the mean of the x_readings )
    b0 = cal_mean(y_readings) - (b1 * cal_mean(x_readings))
 


# In[14]:


def predict_target_value(x, b0, b1):
    """
    Calculating the target (y) value using the input x and the coefficients b0, b1
    :param x:
    :param b0:
    :param b1:
    :return:
    """
    return b0 + b1 * x
 


# In[15]:


def cal_rmse(actual_readings, predicted_readings):
    """
    Calculating the root mean square error
    :param actual_readings:
    :param predicted_readings:
    :return:
    """
    square_error_total = 0.0
    total_readings = len(actual_readings)
    for i in xrange(0, total_readings):
        error = predicted_readings[i] - actual_readings[i]
        square_error_total += pow(error, 2)
    rmse = square_error_total / float(total_readings)
    return rmse
 


# In[17]:


def simple_linear_regression(dataset):
    """
    Implementing simple linear regression without using any python library
    :param dataset:
    :return:
    """
 
    # Get the dataset header names
    dataset_headers = get_headers(df)
    print ("Dataset Headers :: ", dataset_headers)
 
    # Calculating the mean of the square feet and the price readings
    square_feet_mean = cal_mean(df[dataset_headers[0]])
    price_mean = cal_mean(df[dataset_headers[1]])
 
    square_feet_variance = cal_variance(df[dataset_headers[0]])
    price_variance = cal_variance(df[dataset_headers[1]])
 
    # Calculating the regression
    covariance_of_price_and_square_feet = df.cov()[dataset_headers[0]][dataset_headers[1]]
    w1 = covariance_of_price_and_square_feet / float(square_feet_variance)
 
    w0 = price_mean - (w1 * square_feet_mean)
 
    # Predictions
    dataset['Predicted_Price'] = w0 + w1 * df[dataset_headers[0]]


# In[18]:


if __name__ == "__main__":
 
    
    simple_linear_regression(df)


# In[21]:


# Packages for creating the graphs
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.graph_objs import *
#py.sign_in(YOUR_USER_NAME, YOUR_API_KEY)
 
 
def scatter_graph(x, y, graph_title, x_axis_title, y_axis_title):
    """
    Scatter Graph
    :param x: 
    :param y: 
    :param graph_title: 
    :param x_axis_title: 
    :param y_axis_title: 
    :return: 
    """
    trace = go.Scatter(
        x=x,
        y=y,
        mode='markers'
    )
    layout = go.Layout(
        title=graph_title,
        xaxis=dict(title=x_axis_title), yaxis=dict(title=y_axis_title)
    )
    data = [trace]
    fig = Figure(data=data, layout=layout)
    plot_url = py.plot(fig)


# In[27]:


scatter_graph(X.iloc[:,1], Y, "graph_title", "x_axis_title", "y_axis_title")


# In[25]:


df['Predicted_Price']

