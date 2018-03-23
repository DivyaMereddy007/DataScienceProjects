
# coding: utf-8

# In[794]:


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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn import tree 
import pydotplus 
from sklearn.cross_validation import  cross_val_score 
from IPython.display import Image 
from sklearn.externals.six import StringIO 
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.metrics import mean_squared_error 
import statistics as st


# In[897]:


#read CSV files of 2 data sets
my_data = pd.read_csv("C:/Users/mereddda/Desktop/IDA/breast-cancer-wisconsin.csv", skipinitialspace=True)    
my_data.drop(my_data.columns[[1]], axis=1, inplace=True)
my_data.drop(my_data.columns[[0]], axis=1, inplace=True)
my_data.head()


# In[898]:



my_data.head()
X=my_data.iloc[:,1:]
Y=my_data.iloc[:,0]

X_test=X.iloc[:132,:]
X_train=X.iloc[132:,:]
Y_test=Y.iloc[:132]
Y_test
Y_train=Y.iloc[132:]
#Y_train
#X_train


# In[918]:


#wine:
#read CSV files of 2 data sets
my_data = pd.read_csv("C:/Users/mereddda/Desktop/IDA/winequality-white.csv", skipinitialspace=True)    
my_data.head()
train,test=train_test_split(my_data,random_state=20, test_size=0.33)
X_train=train.iloc[:,:-2]
Y_train=train.iloc[:,-1]
X_test=test.iloc[:,:-2]
Y_test=test.iloc[:,-1]


# In[931]:


my_data = pd.read_csv("C:/Users/mereddda/Desktop/IDA/winequality-red.csv", skipinitialspace=True)   
X_test=my_data.iloc[:,:-2]
Y_test=my_data.iloc[:,-1]


# In[919]:


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right


# In[920]:


def df_split(df, feature, value):
 left = df[df[feature]<=value]
 right = df[df[feature]>value]
 return left, right #


# In[921]:


global Count
Count=list()
global MSB_list
MSB_list=list()
# Select the best split point for a dataset
def get_split(dataset):
#    X=dataset.iloc[:,:-2]
#    Y=dataset.iloc[:,-1]
    X=dataset.iloc[:,:-2]
    Y=dataset.iloc[:,-1]
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    print(X);print(Y)
    correlation =X.apply(lambda x: x.corr(Y)) 
    print('correlation'); print(correlation)
    max_col = correlation.idxmax() 
    print(max_col)

    col_med = st.median(dataset.loc[:,max_col]) 
    right = dataset.loc[dataset.loc[:,max_col] >= col_med] 
    left = dataset.loc[dataset.loc[:,max_col] < col_med] 
#    groups=df_split(dataset, max_col, col_med)
#    print(type(groups))
#    tup=(4,5)
#    print(type(tup))
    b_groups=(right,left)
#    dataset = dataset.drop(max_col,1) 
#     return left, right #(left df, right df
    c=(len(left),len(right))
    Count.append(c)
   
#    msb=(MSE_error(left),MSE_error(right))
#    Count.append(msb)
    return {'index':max_col, 'value':col_med, 'groups':b_groups}
#get_split(my_data)


# In[922]:


# Create a terminal node value
def to_terminal(group):
    outcomes = group.loc[:,'quality'].mean()
    return outcomes


# In[923]:


# Create child splits for a node or make terminal
def split(node):
 left, right = node['groups']
 del(node['groups'])
 # check for a no split
 if len(left) == 0 or len(right) == 0 :
   return
 # process left child
 left_MSE = MSE_error(left)
 right_error = MSE_error(right)
 msb=(left_MSE,right_error)
 MSB_list.append(msb)
 if left_MSE <= 500:
   node['left'] = to_terminal(left)
 else:
   node['left'] = get_split(left)
   split(node['left'])
 # process right child
 if right_error <= 500:
   node['right'] = to_terminal(right)
 else:
   node['right'] = get_split(right)
   split(node['right'])


# In[840]:


# Create child splits for a node or make terminal
def split(node, max_depth=4, min_size=1, depth=1):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
#	if not left or not right:
#		node['left'] = node['right'] = to_terminal(left + right)
#		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)


# In[924]:


def build_tree(train):
	root = get_split(train) # best split 
	split(root) # split add to func 
	return root


# In[925]:


test.head()


# In[926]:


# Print a decision tree
def print_tree(node, depth=0):
	if isinstance(node, dict):
		print ((depth, (node['index']+1), node['value']))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('%s[%s]' % ((depth*' ', node)))


# In[927]:


#my_data.drop(my_data.columns[[0]], axis=1, inplace=True)
my_data.head()
tree = build_tree(my_data)
tree


# In[928]:


# Print a decision tree
def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s[X%s < %.3f]' % ((depth*' ', (node['index']), node['value'])))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('%s[%s]' % ((depth*' ', node)))

#print(tree)
print_tree(tree)


# In[929]:


# Make a prediction with a decision tree

def predict(node, row):
#	print(node['right']) 
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
       
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']


# In[937]:


test=X_test
for i in range(len(test)):
	prediction = predict(tree, test.iloc[i])
	print(prediction);


# In[938]:


Count, MSB_list


# In[689]:


dataset = [[2.771244718,1.784783929,0],
	[1.728571309,1.169761413,0],
	[3.678319846,2.81281357,0],
	[3.961043357,2.61995032,0],
	[2.999208922,2.209014212,0],
	[7.497545867,3.162953546,1],
	[9.00220326,3.339047188,1],
	[7.444542326,0.476683375,1],
	[10.12493903,3.234550982,1],
	[6.642287351,3.319983761,1]]
 
#  predict with a stump
stump = {'index': 0, 'right': 1, 'value': 6.642287351, 'left': 0}
for row in dataset:
	prediction = predict(stump, row)
	print(row)
	print('Expected=%d, Got=%d' % (row[-1], prediction))


# In[690]:


y_predict = [y_pred] * len(Y_train)
train_MSE_error = mean_squared_error(Y_train, y_predict) 
train_MSE_error


# In[691]:


y_predict = [y_pred] * len(Y_test) 
test_MSE_error  = mean_squared_error(Y_test, y_predict) 
test_MSE_error


# In[692]:


X_train_sample = X_train;


# In[693]:


correlation = X_train_sample.apply(lambda x: x.corr(Y_train)) 
max_col = correlation.idxmax() 
col_med = st.median(X_train.loc[:,max_col]) 
print(col_med)
datax = np.where(X_train.loc[:,max_col] > col_med)
datax


# In[694]:


X_train_right = X_train.loc[X_train.loc[:,max_col] >= col_med] 
X_train_left = X_train.loc[X_train.loc[:,max_col] < col_med] 
X_train_sample = X_train_sample.drop(max_col,1)
X_train_sample


# In[695]:


def database_split(df):
     correlation =df.apply(lambda x: x.corr(Y_train)) 
     max_col = correlation.idxmax() 
     col_med = st.median(df.loc[:,max_col]) 
     right = df.loc[df.loc[:,max_col] >= col_med] 
     left = df.loc[df.loc[:,max_col] < col_med] 
     df = df.drop(max_col,1) 
     return left, right #(left df, right df
database_split(X_train)    


# In[696]:


def MSE_error(df):
    y_pred = df.mean() 
    y_predict = [y_pred] * len(df)
    train_MSE_error = mean_squared_error(df, y_predict) 
    return train_MSE_error
MSE_error(Y_train)


# In[697]:


left,right=database_split(X_train) 
#database_split(Y_train)
if(database_split())


# In[698]:


right

