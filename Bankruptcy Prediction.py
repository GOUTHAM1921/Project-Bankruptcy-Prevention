#!/usr/bin/env python
# coding: utf-8

# In[3]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[4]:


#Load Data set
Data= pd.read_csv("bankruptcy-prevention.csv",sep=';')


# In[5]:


Data


# #### INDUSTRIAL RISK       -  debt repayment ability of companies in a specific industry .
# 
# 
# #### MANAGEMENT RISK       -  Management risk is the risk—financial, ethical, or otherwise—associated with ineffective, destructive,                                                        or underperforming management.
# 
# #### FINANCIAL FLEXIBILITY - “the ability of a firm to access and restructure its financing at a low cost.” Flexibility lessens the                                                                  underinvestment problems if access to capital is limited and helps to avoid financial distress.
# 
# #### CREDIBILITY                      - the quality of being trusted and believed in.
# 
# #### COMPETITIVENESS          - possession of a strong desire to be more successful than others.
# 
# #### OPERATING RISK             - Operational risk is the risk of losses caused by flawed or failed processes, policies, systems or events                                                       that disrupt business operations. Employee errors, criminal activity such as fraud, and physical events                                                       are among the factors that can trigger operational risk.
# 

# In[6]:


Data.head()


# In[7]:


Data.columns


# # EXPLORATORY DATA ANALYSIS

# In[8]:


#shape
Data.shape


# In[9]:


Data.info()


# In[10]:


Data.describe()


# In[11]:


#data types
Data.dtypes


# In[12]:


Data.nunique() #checking for unique values in each feature


# In[13]:


Data[" class"].value_counts() 


# In[15]:


#check for null values
Data.isnull().sum()


# In[14]:


Data.duplicated().sum() #checking for duplicate values


# # VISUALIZATION

# In[15]:


sns.countplot(x=" class",data=Data,palette="hls")


# In[16]:


sns.catplot(data=Data, orient="h", kind="box")


# In[17]:


Data.plot(kind='hist', subplots=True, layout=(3,3), figsize=(20, 8),sharex=False)
plt.show()


# In[18]:


def countplots(data):
    plt.subplots(3,3, figsize = (20,15))
    i = 1
    for feature in data.columns:
        plt.subplot(4,2,i)
        sns.countplot(data = data, x = feature, hue=' class',palette="hls")
        i+=1 


# In[19]:


countplots(Data)


# In[20]:


#label encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
Data.iloc[:,6]=le.fit_transform(Data.iloc[:,6])
Data.head()


# # CORRELATION ANALYSIS

# In[21]:


Data_corr=Data.corr()
Data_corr


# In[22]:


sns.heatmap(Data_corr)


# # MODEL BUILDING

# In[96]:


X=Data.iloc[:,0:6]
Y=Data.iloc[:,6]


# ### Logistic Regression

# In[97]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split  
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# In[98]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=12)


# In[99]:


classifier = LogisticRegression()
classifier.fit(x_train,y_train)

train_pred=classifier.predict(x_train)
test_pred=classifier.predict(x_test)

log_trainscore=accuracy_score(y_train,train_pred)
log_testscore=accuracy_score(y_test,test_pred)

print(log_trainscore,log_testscore)


# In[100]:


pd.crosstab(y_test,test_pred)


# In[101]:


from sklearn.metrics import classification_report 
print (classification_report (y_test, test_pred))


# In[102]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
Logit_roc_score=roc_auc_score(y_test, test_pred)
Logit_roc_score 


# ### KNN

# In[103]:


import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# In[104]:


n_neighbors=np.array([2*i+1 for i in range(0,20)])
param_grid=dict(n_neighbors=n_neighbors)

KNN=KNeighborsClassifier()
grid=GridSearchCV(estimator=KNN,param_grid=param_grid,cv=10)
grid.fit(X,Y)
KNN_grid=grid.best_score_
KNN_param=grid.best_params_

print(KNN_grid,":",KNN_param)


# ### Decision Tree

# In[105]:


from sklearn import datasets  
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing


# In[109]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=10)
DT_model=DecisionTreeClassifier(criterion='gini',max_depth=3)
DT_model.fit(x_train,y_train)


# In[110]:


predict=DT_model.predict(x_test)
pd.crosstab(y_test,predict)


# In[111]:


DT=np.mean(y_test==predict)
DT


# ### Bagging

# In[36]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier


# In[37]:


kfold=KFold(n_splits=10)
num_trees=100
Bag_model=BaggingClassifier(max_samples=0.8,n_estimators=num_trees,random_state=8)
results=cross_val_score(Bag_model,X,Y,cv=kfold)
BAG=results.mean()
BAG


# In[38]:


from sklearn.model_selection import cross_val_predict
bag_pred = cross_val_predict(model, X,Y, cv=kfold)
pd.crosstab(Y,bag_pred)


# In[40]:


np.mean(bag_pred==Y)


# ### Random Forest

# In[112]:


from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import  RandomForestClassifier

from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing


# In[113]:


k_range=[10,20,50,100,200,300,400,500]
k_scores=[]
max_features=5
for k in k_range:
    Ran_model=RandomForestClassifier(n_estimators=k,max_samples=0.8,max_features=max_features,random_state=8)
    results=cross_val_score(Ran_model,X,Y,cv=10)
    k_scores.append(results.mean())
    
k_scores


# In[114]:


kfold=KFold(n_splits=10)
num_trees=100
max_features=5
Ran_model=RandomForestClassifier(n_estimators=num_trees,max_samples=0.8,max_features=max_features,random_state=8)
results=cross_val_score(Ran_model,X,Y,cv=kfold)
RF=results.mean()
RF


# In[115]:


from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(model, X,Y, cv=kfold)
pd.crosstab(Y,y_pred)


# ### XG Boost

# In[53]:


get_ipython().system('pip install xgboost')


# In[55]:


from xgboost import XGBClassifier


# In[56]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=12)


# In[116]:


XG_model = XGBClassifier()
XG_model.fit(x_train, y_train)


# In[117]:


y_pred = model.predict(x_test)
XG_accuracy = accuracy_score(y_test, y_pred)
XG_accuracy


# ### Adaboost

# In[54]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

kfold=KFold(n_splits=10)
num_trees=100
Boost_model=AdaBoostClassifier(n_estimators=num_trees,learning_rate=0.8,random_state=8)
results=cross_val_score(Boost_model,X,Y,cv=kfold)
BOOST=results.mean()
BOOST


# ### SVM

# In[60]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix


# In[62]:


train_X,test_X,train_Y,test_Y=train_test_split(X,Y,test_size=0.3,random_state=10)


# In[63]:


model_linear=SVC(kernel='linear')
model_linear.fit(train_X,train_Y)

train_pred_lin=model_linear.predict(train_X)
test_pred_lin=model_linear.predict(test_X)

train_lin_acc=np.mean(train_pred_lin==train_Y)
test_lin_acc=np.mean(test_pred_lin==test_Y)

SVM_LINEAR=test_lin_acc
SVM_LINEAR


# In[64]:


model_rbf=SVC(C=15,gamma=0.0001,kernel='rbf')
model_rbf.fit(train_X,train_Y)

train_pred_rbf=model_linear.predict(train_X)
test_pred_rbf=model_linear.predict(test_X)

train_rbf_acc=np.mean(train_pred_lin==train_Y)
test_rbf_acc=np.mean(test_pred_lin==test_Y)


SVM_RBF=test_rbf_acc
SVM_RBF


# ### Naive Bayes

# In[65]:


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB


# In[66]:


train_X,test_X,train_Y,test_Y=train_test_split(X,Y,test_size=0.3,random_state=10)


# In[67]:


Gmodel=GaussianNB()

Gmodel.fit(train_X,train_Y)
NB2=Gmodel.predict(test_X)
G_acc=np.mean(NB2==test_Y)
G_acc


# In[68]:


Mmodel = MultinomialNB()

Mmodel.fit(train_X,train_Y)
MB = Mmodel.predict(test_X)
M_acc = np.mean(MB==test_Y) 
M_acc


# ### Neural Networks

# In[88]:


get_ipython().system('pip install tensorflow')
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation,Layer,Lambda


# In[89]:


train_X,test_X,train_Y,test_Y=train_test_split(X,Y,test_size=0.3,random_state=10)


# In[90]:


def prep_model(hidden_dim):
    model = Sequential()
    for i in range(1,len(hidden_dim)-1):
        if (i==1):
            model.add(Dense(hidden_dim[i],input_dim=hidden_dim[0],activation="relu"))
        else:
            model.add(Dense(hidden_dim[i],activation="relu"))
    
    model.add(Dense(hidden_dim[-1],activation="sigmoid"))
    model.compile(loss="binary_crossentropy",optimizer = "adam",metrics = ["accuracy"])
    return model 


# In[125]:


first_model = prep_model([6,9,6,1])
first_model.fit(train_X,train_Y,validation_split=0.33,epochs=100,batch_size=30)


# In[128]:


scores=first_model.evaluate(np.array(train_X),np.array(train_Y))
print("%s: %.2f%%" % (first_model.metrics_names[1], scores[1]*100))


# In[129]:


first_model.fit(test_X,test_Y,validation_split=0.33,epochs=100,batch_size=30)


# In[130]:


scores2=first_model.evaluate(np.array(test_X),np.array(test_Y))
print("%s: %.2f%%" % (first_model.metrics_names[1], scores2[1]*100))


# ## Accuracy of all Models

# In[137]:


my_dict={"Model":["Log_Reg","KNN","DecisionTree","Bagging","RandomForest","XGBoost","ADABOOST","SVM_Linear","SVM_RBF","NB_Gaussian",'NB_Multinomial',"Neural_Network"],
         "Test_Accuracy":[log_testscore,KNN_grid,DT,BAG,RF,XG_accuracy,BOOST,SVM_LINEAR,SVM_RBF,G_acc,M_acc,scores2]}
DF=pd.DataFrame(my_dict)
DF


# ### Dumping the File

# In[138]:


import joblib
import pickle
from pickle import dump
from pickle import load


# In[140]:


dump(classifier,open('Log_Reg.sav','wb'))

