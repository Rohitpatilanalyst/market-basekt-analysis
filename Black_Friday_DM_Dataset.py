#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#To visualize the whole grid
pd.options.display.max_columns = 999


# In[2]:


blackfriday=pd.read_csv("/Users/bilge/Downloads/BlackFriday.csv")


# In[3]:


blackfriday.isnull().sum()


# In[4]:


#here we are making an array consisting of 2 columns namely Product_Category_2,Product_Category_3
b = ['Product_Category_2','Product_Category_3'] 


# In[5]:


for i in b:
    exec("blackfriday.%s.fillna(blackfriday.%s.value_counts().idxmax(), inplace=True)" %(i,i))


# In[6]:


X = blackfriday#.drop(["Purchase"], axis=1)


# In[7]:


from sklearn.preprocessing import LabelEncoder #import encoder from sklearn library
LE = LabelEncoder()
#Now we will encode the data into labels using label encoder for easy computing


# In[8]:


X = X.apply(LE.fit_transform)


# In[9]:


X.Gender = pd.to_numeric(X.Gender)
X.Age = pd.to_numeric(X.Age)
X.Occupation = pd.to_numeric(X.Occupation)
X.City_Category = pd.to_numeric(X.City_Category)
X.Stay_In_Current_City_Years = pd.to_numeric(X.Stay_In_Current_City_Years)
X.Marital_Status = pd.to_numeric(X.Marital_Status)
X.Product_Category_1 = pd.to_numeric(X.Product_Category_1)
X.Product_Category_2 = pd.to_numeric(X.Product_Category_2)
X.Product_Category_3 = pd.to_numeric(X.Product_Category_3)


# In[10]:


numeric_features = X.select_dtypes(include=[np.number])
numeric_features.dtypes


# In[11]:


corr = numeric_features.corr()
print (corr['Purchase'].sort_values(ascending=False)[:12], '\n')
print (corr['Purchase'].sort_values(ascending=False)[-12:])


# In[12]:


#correlation matrix
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corr, vmax=.8,annot_kws={'size': 20}, annot=True);


# In[ ]:





# In[13]:


# feature representing the count of each user
def getCountVar(compute_df, count_df, var_name):
	grouped_df = count_df.groupby(var_name)
	count_dict = {}
	for name, group in grouped_df:
		count_dict[name] = group.shape[0]

	count_list = []
	for index, row in compute_df.iterrows():
		name = row[var_name]
		count_list.append(count_dict.get(name, 0))
	return count_list


# In[14]:


#data[“User_ID_Count”] = getCountVar(data, data, “User_ID”)
X["Product_Category_1_Count"] =getCountVar(X, X,"Product_Category_1")
X["Product_Category_2_Count"] =getCountVar(X, X, "Product_Category_2")
X["Product_Category_3_Count"] =getCountVar(X, X,"Product_Category_3")
X["Product_ID_Count"] =getCountVar(X, X, "Product_ID")


# In[15]:


corr = numeric_features.corr()
print (corr['Purchase'].sort_values(ascending=False)[:30], '\n')
print (corr['Purchase'].sort_values(ascending=False)[-30:])


# In[16]:


corr = numeric_features.corr()
print (corr['Purchase'].sort_values(ascending=False)[:30], '\n')
print (corr['Purchase'].sort_values(ascending=False)[-30:])


# In[ ]:





# In[17]:


X = X.drop(["Purchase"], axis=1)


# In[18]:


X = X.drop(['User_ID'],axis=1)


# In[19]:


X = X.drop(['Product_ID'],axis=1)


# In[20]:


from sklearn.preprocessing import LabelEncoder#Now let's import encoder from sklearn library
LE = LabelEncoder()
#Now we will encode the data into labels using label encoder for easy computing


# In[21]:


X = X.apply(LE.fit_transform)#Here we applied encoder onto data


# In[22]:


Y = blackfriday["Purchase"]#Here we will made a array named as Y consisting of data from purchase column


# In[23]:


from sklearn.preprocessing import StandardScaler
SS = StandardScaler()


# In[24]:


Xs = SS.fit_transform(X)
#You must to transform X into numeric representation (not necessary binary).Because all machine learning methods operate on matrices of number


# In[25]:


from sklearn.decomposition import PCA
pc = PCA(6)#here 6 indicates the number of components you want it into.


# In[26]:


principalComponents = pc.fit_transform(X)#Here we are applying PCA to data/fitting data to PCA


# In[27]:


pc.explained_variance_ratio_


# In[28]:


principalDf = pd.DataFrame(data = principalComponents, columns = ["component 1", "component 2", "component 3", "component 4","component 5",
                                                                  "component 6"])


# In[29]:


from sklearn.model_selection import KFold
kf = KFold(20)
#Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (without shuffling by default).
#Each fold is then used once as a validation while the k - 1 remaining folds form the training set.


# In[30]:


for a,b in kf.split(principalDf):
    X_train, X_test = Xs[a],Xs[b]
    y_train, y_test = Y[a],Y[b]


# In[31]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[32]:


lr = LinearRegression()
dtr = DecisionTreeRegressor()
rfr = RandomForestRegressor()
gbr = GradientBoostingRegressor()


# In[33]:


fit1 = lr.fit(X_train,y_train)#Here we fit training data to linear regressor
fit2 = dtr.fit(X_train,y_train)#Here we fit training data to Decision Tree Regressor
fit3 = rfr.fit(X_train,y_train)#Here we fit training data to Random Forest Regressor
fit4 = gbr.fit(X_train,y_train)#Here we fit training data to Gradient Boosting Regressor


# In[34]:


print("Accuracy Score of Linear regression on train set",fit1.score(X_train,y_train)*100)
print("Accuracy Score of Decision Tree on train set",fit2.score(X_train,y_train)*100)
print("Accuracy Score of Random Forests on train set",fit3.score(X_train,y_train)*100)
print("Accuracy Score of Gradient Boosting on train set",fit4.score(X_train,y_train)*100)


# In[35]:


print("Accuracy Score of Linear regression on test set",fit1.score(X_test,y_test)*100)
print("Accuracy Score of Decision Tree on test set",fit2.score(X_test,y_test)*100)
print("Accuracy Score of Random Forests on test set",fit3.score(X_test,y_test)*100)
print("Accuracy Score of Gradient Boosting on testset",fit4.score(X_test,y_test)*100)


# In[ ]:




