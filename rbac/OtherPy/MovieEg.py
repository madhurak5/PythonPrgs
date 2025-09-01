"""
import pandas as pd
data1 = pd.read_csv("D:/PythonPrgs/car.csv")
print("The columns in the dataset are : \n", data1.keys())
print("No. of rows and columns in the dataset : ", data1.shape)
# Alternately
attrList = data1.keys()

print("Training Set : \n")
data1_x = pd.DataFrame(data1, columns=attrList)
print(data1_x)

print ("Testing Set : \n")
data1_y = pd.DataFrame(data1, columns=data1['unacc'])
print(data1_y.keys())
print("Statistical descriptions : \n",data1.describe())
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# load tic-tac-toe data
data = pd.read_csv("tic-tac-toe.data", sep=",")
data.rename(columns={'x': 'top left', 'x.1': 'top middle', 'x.2': 'top right','x.3': 'middle left', 'o': 'middle middle', 'o.1' : 'middle right', 'x.4' : 'bottom left', 'o.2' : 'bottom middle', 'o.3':'bottom right','positive' : 'outcome'},inplace=True)
data.head(3)
data_final = pd.concat([data_new, data.ix[:,9]],axis=1)
# Split data into training and test sets
train, test = train_test_split(data_final, test_size=0.3)
train.head()
#Applying logistic Regression model
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
x_train = train.ix[:, :-1]
y_train= train.ix[:, -1]
x_test = test.ix[:, :-1]
y_test= test.ix[:, -1]
LR = LR.fit(x_train, y_train)
LR.score(x_train, y_train)
#Evaluate model on test data
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
prob = LR.predict_proba(x_test)
predicted= LR.predict(x_test)
#Findind the accuracy using confusion matrix
print(metrics.accuracy_score(y_test, predicted))
print(metrics.confusion_matrix(y_test, predicted))
print(metrics.classification_report(y_test, predicted))
#Finding the accuracy using cross validation method
scores = cross_val_score(LogisticRegression(), data_final.ix[:, 0:27], data_final.ix[:, 27], scoring='accuracy', cv=10)
print(scores)
print(scores.mean())
