#!/usr/bin/env python
# coding: utf-8

# In[24]:


#import the library
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from sklearn.decomposition import PCA


# In[28]:


#load the seeds file path and convert it to dataframe
d= pd.read_csv(r"M:\Lab assignment (unsupervised) material-20221130\seeds.csv")

#Drop the Type(target) column
ddd=d.drop(['Type'], axis=1)

#Standardization of data to mean of zero and std of one
Xm = np.mean(ddd,axis=0)
Xs = np.std(ddd,axis=0)
dd = (ddd-Xm)/(Xs)
print(dd)


# In[35]:


##Decomposition of original data 

#Extracted U of SVD from data   e.g    A.(A^T)
Ui = dd.dot(dd.transpose())

#Extracted V of SVD from data   e.g    (A^T).A 
Vi = dd.transpose().dot(dd)


#decomposition to calculate eigen values
eig_values, U = np.linalg.eig(Ui)              
U = U.real
eig_values.real  #Converting the complex values to real values
print(U.shape)
   
#decomposition to calculate eigen values
V_eig_values, V = np.linalg.eig(Vi)
V = V.real
V_eig_values.real  #Converting the complex values to real values
print(V.shape)


# In[36]:


#Making diagonal matrix with the diagonal elements as the eigen values
Si = np.diag(V_eig_values)  
Si = np.sqrt(Si)
n = np.zeros(shape = (192, 7), dtype = int)
Si=np.append(Si, n, axis = 0)
print(Si.shape)

#Considering first two principal components(columns) only
n_elements = 2
Si = Si[:, :n_elements]
V = V[:n_elements, :]
Sii=Si[0:2]


# In[38]:


#principal components
pcs = U @ Si               
print(pcs.shape)

#first principal component
pc1 = pcs[:, 0]          

#second principal component
pc2 = pcs[:, 1]

#making dataframe of principal components
principal_comp = pd.DataFrame(data=pcs , columns = ['PC1','PC2'])

#appending target column also
principal_df = pd.concat([principal_comp , d[['Type']]], axis = 1)
print(principal_df)


# In[40]:


#Using sklearn.decomposition.pca for comparison

df=dd.to_numpy()
pca1 = PCA(n_components=2)
projec=pca1.fit_transform(df)

# shows the variance that can be attributed to each of the principal components
#print(pca1.explained_variance_ratio_)           
#print(pca1.singular_values_)

#making dataframe of two principal components
pri = pd.DataFrame(data=projec , columns = ['PC_1','PC_2'])
pri=pri.mul({'PC_1': 1, 'PC_2': -1})
pri = pd.DataFrame(data=pri , columns = ['PC_1','PC_2'])

#appending target column also
princ = pd.concat([pri , d[['Type']]], axis = 1)
print(princ)


# In[42]:


#plotting from sklearn.decomposition.pca command

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title(' PCA by library', fontsize = 20)
targets = [1,2,3]
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = princ['Type'] == target
    ax.scatter(princ.loc[indicesToKeep, 'PC_1']
               , princ.loc[indicesToKeep, 'PC_2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


#Plotting from self made PCA from scratch using SVD
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title(' PCA from scratch', fontsize = 20)
targets = [1,2,3]
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = principal_df['Type'] == target
    ax.scatter(principal_df.loc[indicesToKeep, 'PC1']
               , principal_df.loc[indicesToKeep, 'PC2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

