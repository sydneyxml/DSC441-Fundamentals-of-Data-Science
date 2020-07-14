
# coding: utf-8

# In[1]:

## Import warnings. Supress warnings (for  matplotlib)
import warnings
warnings.filterwarnings("ignore")


# In[2]:

## Import analysis modules
import pandas as p
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, roc_curve, auc


## Import Scikit Learn KNN
from sklearn.neighbors import KNeighborsClassifier

## Import visualization modules
import matplotlib.pyplot as plt


# In[3]:

## Read in file
data = p.read_csv('/home/avelkoski/backup-20191201/backup-20190404/Teaching/DSC441/2020-Summer/Labs/knn/final.csv',delimiter=',',na_values='nan')


# In[4]:

## Count of instances and features
rows, columns = data.shape
print (data.shape)


# In[5]:

## Instantiate class
mm = MinMaxScaler()


# In[6]:

## Scale annual income
data['annual_inc'] = mm.fit_transform(data['annual_inc'].values.reshape(-1, 1))


# In[7]:

## Seperate input features from target feature
x = data['annual_inc'].values.reshape(-1, 1)
y = data['loan_status'].as_matrix()


# In[8]:

## Take a look at x
x


# In[9]:

## Take a look at y
y


# In[10]:

## Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.3,random_state=1)


# In[11]:

## Take a look at the shape
x_train.shape, y_train.shape


# In[12]:

## Create the estimator
knn = KNeighborsClassifier(n_neighbors=3)


# In[13]:

## Fit the model
knn.fit(x_train,y_train)


# In[14]:

## Check accuracy score
print ('%0.2f' % knn.score(x_test,y_test))


# In[15]:

## Run 10 fold cross validation
cvs = cross_val_score(knn,x,y,cv=10)


# In[16]:

## Show cross validation scores
cvs


# In[17]:

## Show cross validation score mean and std
print ('%0.2f, %0.2f' % (cvs.mean(), cvs.std()))


# In[18]:

## Check graph ("mean decrease impurity")
knn.kneighbors_graph(x[0:5]).toarray()


# In[19]:

## Predict y given test set
predictions = knn.predict(x_test)


# In[20]:

## Take a look at the confusion matrix ([TN,FN],[FP,TP])
confusion_matrix(y_test,predictions)


# In[21]:

## Accuracy score
print ('%0.2f' % precision_score(y_test, predictions))


# In[22]:

## Recall score
print ('%0.2f' % recall_score(y_test, predictions))


# In[23]:

## Print classification report
print (classification_report(y_test, predictions))


# In[24]:

## Get data to plot ROC Curve
fp, tp, th = roc_curve(y_test, predictions)
roc_auc = auc(fp, tp)


# In[25]:

## Plot ROC Curve
plt.title('ROC Curve')
plt.plot(fp, tp, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

