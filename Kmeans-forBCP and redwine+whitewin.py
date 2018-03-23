
# coding: utf-8

# In[1]:


import pandas as pd
import random
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import scipy.cluster.hierarchy as hac
from sklearn import metrics
from matplotlib import style
from itertools import groupby
from scipy.special import comb
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import accuracy_score
#read CSV files of 2 data sets


# In[2]:


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


# In[3]:


StudentData=pd.read_csv("C:/Users/mereddda/Desktop/IDA/winequality-red - Copy.csv", skipinitialspace=True)  


# In[748]:


my_data2 = pd.read_csv("C:/Users/mereddda/Desktop/IDA/breast-cancer-wisconsin.csv", skipinitialspace=True)    
my_data2.head()


# In[747]:


my_data1=my_data2.drop(my_data2.columns[[0,2]], axis=1)
my_data1.head()


# In[710]:


my_data=my_data1
my_data.head()


# In[711]:


my_data['quality'] = my_data['quality'].replace({'R': 1, 'N': 0})

Class_Data=my_data['quality']
#my_data.drop(my_data.columns[[0]], axis=1, inplace=True)

features = list(my_data.columns[1:])
StudentData=my_data[features]
#Y_train
#X_train
#bool_cols = df.columns[df.dtypes == 'bool']
#my_data.head()


# In[712]:



#StudentData['Outcome'] = StudentData['Outcome'].replace({'R': 1, 'N': 0})


# In[758]:


StudentData.head()


# In[12]:


def kmeans_students(nclusters,fulldata) :
	kmeans = KMeans(n_clusters=nclusters)
	kmeans.fit(fulldata)
	global centroids
	centroids= kmeans.cluster_centers_
	labels_k = kmeans.labels_
	error_value = kmeans.inertia_
	l_error_values.append(error_value)
	print ("Centroid for K-means clustering with k-value of ",nclusters,": \n",centroids)
	print ("SSE value with k-value of ", nclusters,": ",error_value)
	print ("Labels with k-value of ", nclusters,": ",labels_k)
	population = np.bincount(labels_k[labels_k>=0])
	print ("Population of the cluster with k-value of", nclusters,": ",population)
#	K_1_label=population.argmax();print("Highest Population of the cluster with k-value of", nclusters,": ",K_1_label)
#    a.argmax()
	#for j in len(centroids):
	  #print "SSE for each cluster: ", nclusters.inertia
	#plt.scatter(nclusters, error_value)
	sc_score = metrics.silhouette_score(fulldata,labels_k,metric='euclidean',sample_size=len(StudentData['quality']))
	print ("Silhouette coefficient score with k-value of ",nclusters,": ",sc_score,"\n\n")
	plt2.scatter( nclusters,error_value)
#	print('calling SSE-Divya alert')    
	SSE0,SSE1,SSE2,SSE3=Cal_SSE(fulldata,labels_k,centroids)  
	print('SSE0:',SSE0)
	print('SSE1:',SSE1)
	print('SSE2:',SSE2)
	print('SSE2:',SSE3)    
	return labels_k, sc_score, error_value,kmeans


# In[6]:


StudentData.head()


# In[13]:


l_error_values, labels_k_list, sc_score_list, sse_list,labels_k,kmeans_list = [], [], [], [],[],[]
nclusters =[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
for i in nclusters:
    labels_k, sc_score, sse,kmeans = kmeans_students(i,StudentData)
    sc_score_list.append(sc_score)
    sse_list.append(sse)
    labels_k_list.append(labels_k)
    kmeans_list.append(kmeans)


# In[735]:


j=0
sc_diff, sse_diff = [],[]
for i in range(0,len(nclusters)) :
 if i != 0 :
    sc_diff.append(abs(sc_score_list[j]-sc_score_list[i]))
    sse_diff.append(abs(sse_list[j]-sse_list[i]))
    j = j + 1
i = i + 1

print (sse_list)
print (sc_score_list)
print(labels_k_list)
Labels_k_list_final=labels_k_list[2]


# In[736]:


print("The difference in the Silhouette Coefficient in each of the clusters: ",sc_diff)
print("The difference in the SSE Value in each of the clusters: ", sse_diff)
print("Best k-value with Silhouette Coefficient Score is: ",nclusters[sc_diff.index(max(sc_diff))])
print("Best k-value with SSE is: ",sse_diff.index(max(sse_diff)))


# In[742]:


#why are we adjusting the data
labels_k, sc_score, error_value,kmeans = kmeans_students(nclusters[sc_diff.index(max(sc_diff))],StudentData[features])
adjusted_labels_k = [x+1 for x in labels_k]
#plt.show()
plt2.show()


# In[738]:


def kmeans_students(nclusters,fulldata) :
	kmeans = KMeans(n_clusters=nclusters)
	kmeans.fit(fulldata)
	centroids = kmeans.cluster_centers_
	labels_k = kmeans.labels_
	error_value = kmeans.inertia_
	l_error_values.append(error_value)
	print ("Centroid for K-means clustering with k-value of ",nclusters,": \n",centroids)
	print ("SSE value with k-value of ", nclusters,": ",error_value)
	print ("Labels with k-value of ", nclusters,": ",labels_k)
	population = np.bincount(labels_k[labels_k>=0])
	print ("Population of the cluster with k-value of", nclusters,": ",population)
#	K_1_label=population.argmax();print("Highest Population of the cluster with k-value of", nclusters,": ",K_1_label)
#    a.argmax()
	#for j in len(centroids):
	  #print "SSE for each cluster: ", nclusters.inertia
	#plt.scatter(nclusters, error_value)
	sc_score = metrics.silhouette_score(fulldata,labels_k,metric='euclidean',sample_size=len(StudentData['Radius_Mean']))
	print ("Silhouette coefficient score with k-value of ",nclusters,": ",sc_score,"\n\n")
	plt2.scatter(nclusters,sc_score)
	return labels_k, sc_score, error_value,kmeans


# In[720]:


array_2=Labels_k_list_final[Labels_k_list_final==0]
StudentData_1=my_data[Labels_k_list_final==0]
StudentData_2=my_data[Labels_k_list_final==1]
StudentData_3=my_data[Labels_k_list_final==2]
StudentData_4=my_data[Labels_k_list_final==3]

Y_data=my_data['Outcome'];cluster=[]

if len(StudentData_1[StudentData_1['Outcome']==0])>(len(StudentData_1)/2):
    my_data['Outcome'][Labels_k_list_final==0] = my_data['Outcome'].replace({0: 1})
    cluster.append(1)
else:
    my_data['Outcome'][Labels_k_list_final==0] = my_data['Outcome'].replace({1: 0})
    cluster.append(0)
my_data['Outcome'][Labels_k_list_final==0]

if len(StudentData_2[StudentData_2['Outcome']==1])>(len(StudentData_2)/2):
    my_data['Outcome'][Labels_k_list_final==1] = my_data['Outcome'].replace({0: 1})
    cluster.append(1)
else:
    my_data['Outcome'][Labels_k_list_final==1] = my_data['Outcome'].replace({1: 0})
    cluster.append(0)
my_data['Outcome'][Labels_k_list_final==1]

if len(StudentData_3[StudentData_3['Outcome']==1])>(len(StudentData_3)/2):
    my_data['Outcome'][Labels_k_list_final==2] = my_data['Outcome'].replace({0: 1})
    cluster.append(1)
else:
    my_data['Outcome'][Labels_k_list_final==2] = my_data['Outcome'].replace({1: 0})
    cluster.append(0)
my_data['Outcome'][Labels_k_list_final==2]    

if len(StudentData_4[StudentData_4['Outcome']==1])>(len(StudentData_4)/2):
    my_data['Outcome'][Labels_k_list_final==3] = my_data['Outcome'].replace({0: 1})
    cluster.append(1)
else:
    my_data['Outcome'][Labels_k_list_final==3] = my_data['Outcome'].replace({1: 0})
    cluster.append(0)
my_data['Outcome'][Labels_k_list_final==3]
print('Cluster-1- for N : ', len(StudentData_1[StudentData_1['Outcome']==0]))
print('Cluster-1- for R : ', (len(StudentData_1))-len(StudentData_1[StudentData_1['Outcome']==0]))
print('Cluster-2- for N : ', len(StudentData_2[StudentData_2['Outcome']==0]))
print('Cluster-2- for R : ', (len(StudentData_2))-len(StudentData_2[StudentData_2['Outcome']==0]))
print('Cluster-3- for N : ', len(StudentData_3[StudentData_3['Outcome']==0]))
print('Cluster-3- for R : ', (len(StudentData_3))-len(StudentData_3[StudentData_3['Outcome']==0]))
print('Cluster-4- for N : ', len(StudentData_4[StudentData_4['Outcome']==0]))
print('Cluster-4- for R : ', (len(StudentData_4))-len(StudentData_4[StudentData_4['Outcome']==0]))


cluster


# In[721]:


my_data.head()


# In[722]:


kmeans=kmeans_list[2]
kmeans


# In[723]:


my_data1[features].head()


# In[724]:


#Database=my_data1[features]
#test_data = pd.read_csv("C:/Users/mereddda/Desktop/IDA/breast-cancer-wisconsin.csv", skipinitialspace=True)  
#Database['Outcome'] = Database['Outcome'].replace({'R': 1, 'N': 0})
Test_X=my_data.iloc[:,1:]
Predicted=kmeans.predict(Test_X)
Predicted


# In[725]:


cluster


# In[726]:


Y_predicted=Predicted.copy()
Y_predicted[Predicted==0]=cluster[0]
Y_predicted[Predicted==1]=cluster[1]
Y_predicted[Predicted==2]=cluster[2]
Y_predicted[Predicted==3]=cluster[3]


# In[727]:


Y_predicted


# In[728]:


Y_test=[]
Y_test=my_data['Outcome']
Y_test


# In[729]:


Accuracy = accuracy_score(my_data['Outcome'], Y_predicted)*100
print("Accuracy: ",Accuracy)

def precision_score(y_true, y_pred):
    i = set(y_true).intersection(y_pred)
    len1 = len(y_pred)
    if len1 == 0:
        return 0
    else:
        return len(i) / len1


def recall_score(y_true, y_pred):
    i = set(y_true).intersection(y_pred)
    return len(i) / len(y_true)
Recall1 = (recall_score(my_data['Outcome'], Y_predicted)*100)
Precision1= (precision_score(my_data['Outcome'], Y_predicted)*100)
print("Precision: ",Recall1)
print("recall: ",Precision1)


# In[730]:


a = np.random.rand(3,3)


# In[27]:


a


# In[46]:


a.max()


# In[41]:


a.index(a.max())


# In[51]:


length(a[a==TRUE])


# In[243]:


from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0],
               [4, 2], [4, 4], [4, 0]])

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
kmeans.labels_


# In[244]:



kmeans.predict([[0, 0], [4, 4]])
array([1, 0, 0, 3, 0, 3, 1, 3, 3, 3, 1, 1, 3, 3, 3, 1, 3, 2, 0, 1, 3, 1,
       1, 1, 3, 1, 0, 1, 1, 3, 3, 3, 3, 3, 1, 3, 3, 1, 3, 3, 0, 3, 3, 3,
       3, 1, 3, 0, 1, 3, 0, 3, 3, 3, 3, 1, 1, 1, 0, 1, 1, 1, 0, 3, 3, 1,
       3, 1, 1, 0, 2, 1, 1, 3, 3, 2, 0, 1, 3, 1, 3, 3, 3, 3, 0, 3, 0, 3,
       3, 0, 1, 3, 3, 0, 0, 3, 3, 1, 0, 2, 1, 1, 1, 0, 3, 0, 3, 1, 3, 0,
       1, 3, 1, 2, 0, 1, 0, 0, 3, 3, 0, 0, 1, 0, 0, 1, 1, 0, 2, 1, 3, 2,
       3, 0, 0, 0, 0, 3, 3, 1, 0, 1, 3, 1, 1, 1, 1, 3, 1, 1, 1, 0, 1, 1,
       3, 0, 1, 1, 1, 0, 3, 2, 3, 3, 1, 0, 2, 1, 3, 0, 0, 0, 1, 3, 1, 0,
       1, 1, 0, 3, 3, 1, 3, 0, 1, 3, 2, 2, 3, 0, 0, 1, 3, 0, 3, 1, 0, 1])


# In[368]:


Labels_k_list_final


# In[452]:


my_data['Outcome']


# In[599]:


centroids


# In[4]:


def Cal_SSE(data,k_labels,centroid):
    SSE0,SSE1,SSE2,SSE3=0,0,0,0;
    for label in k_labels:
        if label==0:
            for j in range(0,len(StudentData_1)):
                for i in range(0,12):
                    SSE0=SSE0+( StudentData_1.iloc[j,i]-centroid[0][i])**2
        elif label==1:
            for j in range(0,len(StudentData_2)):
                for i in range(0,12):
                    SSE1=SSE1+( StudentData_2.iloc[j,i]-centroid[1][i])**2
    
        elif label==2:
            for j in range(0,len(StudentData_3)):
                for i in range(0,12):
                    SSE2=SSE2+( StudentData_3.iloc[j,i]-centroid[2][i])**2
        else:
            for j in range(0,len(StudentData_4)):
                for i in range(0,12):
                    SSE3=SSE3+( StudentData_4.iloc[j,i]-centroid[3][i])**2
    return SSE0,SSE1,SSE2,SSE3    
    


# In[5]:


Cal_SSE(my_data,labels_k,centroids)


# In[667]:


StudentData_1.iloc[1,30]


# In[655]:


StudentData_1.iloc[2,8]

