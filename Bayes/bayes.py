
# coding: utf-8

# In[1]:

## Import warnings. Supress warnings (for  matplotlib)
import warnings
warnings.filterwarnings("ignore")


# In[2]:

## Import analysis modules
import pandas as p
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import BernoulliNB


# In[3]:

## Import the data
data = p.read_csv('/home/avelkoski/backup-20191201/backup-20190404/Teaching/DSC441/2020-Summer/Labs/bayes/computer.csv',delimiter=',',na_values='nan')


# In[4]:

data


# In[5]:

## Scikit learn estimators require numeric features
age = {'<=30':0,'31 to 40':1,'>40':2}
income = {'low':0,'medium':1,'high':2}
student = {'no':0,'yes':1}
credit = {'fair':0,'excellent':1}
buys = {'no':0,'yes':1}


# In[6]:

## Convert categorical features to numeric using mapping function
data['age'] = data['age'].map(age)
data['income'] = data['income'].map(income)
data['student'] = data['student'].map(student)
data['credit_rating'] = data['credit_rating'].map(credit)
data['buys_computer'] = data['buys_computer'].map(buys)

data

# In[7]:

x = data.drop('buys_computer',1).as_matrix()
y = data['buys_computer'].as_matrix()


# In[8]:

ohe = OneHotEncoder()


# In[9]:

x = ohe.fit_transform(x).toarray()


# In[10]:

x


# In[11]:

y


# In[12]:

nb = BernoulliNB(alpha=1.0,fit_prior=True)


# In[13]:

nb.fit(x,y)


# In[14]:

x1 = ohe.transform([[0,2,0,1]]).toarray()


# In[15]:

nb.predict(x1)


# In[16]:

x2 = ohe.transform([[0,1,1,0]]).toarray()


# In[17]:

nb.predict(x2)
