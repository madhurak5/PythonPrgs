import pandas as pd
import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
# import tensorflow

# import keras

# from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
# The data is fetched
myData = pd.read_csv("D:/PythonPrgs/titanic.csv")
print (myData.shape)
print(myData.head())

noOfAttr = len(myData.keys())
attr = myData.keys()
print("No. of attributes : ", noOfAttr)
print("Attributes are : ",attr)
for i in range(0, noOfAttr):
    print(attr[i])
print(attr)
# for i in range
print(myData.values) # Prints the data values
# row1 = sum(1 for row in myData)
print("Shape of the dataframe : ", myData.shape)
print("No. of Rows in the dataframe : ",myData.shape[0])
print("No. of Columns in the dataframe : ",myData.shape[1])
#
myData1 = pd.DataFrame(myData, columns=attr) # Creates the dataframe along with the column/feature names
print(myData1.head()) # Prints the first 5 rows from the dataframe
#
print(myData1.Pclass) # Prints the unique names in the column ‘target’
print("With Condition : \n",myData1[myData1.Embarked == 'S'], myData1[myData1.Fare > 100.00], myData1[myData1.Pclass == 1])  # Prints the dataframe where the target value is 0
# df['Wine Name'] = df.target.apply(lambda x:wine.target_names[x])  # Creates a new column ‘Flower Name’ and appends it to the dataframe. Lambda function helps to create or generate another column by applying it to the target column.
# # print(df)

#
plt.xlabel('Fare')   # In the Scatter plot, the x-axis is labeled
plt.ylabel('Pclass')   # In the Scatter plot, the y-axis is labeled
plt.scatter(myData1['Fare'], myData1['Pclass'], color = 'green', marker = '+')
# plt.scatter(df1['alcohol'], df1['malic_acid'], color = 'red', marker = '+')
# plt.scatter(df2['alcohol'], df2['malic_acid'], color = 'blue', marker = '+')
# # plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color = 'blue', marker = '*')
# # plt.scatter(df2['sepal length (cm)'], df2['sepal width (cm)'], color = 'red', marker = "+")
plt.show()
# #
# X = df.drop(['target', 'Wine Name'],axis='columns')  # Creates dataframe, X, by dropping the columns, ‘target’ and ‘Flower Name’
# # print(X.head())
# y = df.target
# print(y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # Splits the data into training and test datasets, 20% as testing set
# print(len(X_train)) # Prints the no. of samples in the training set
# print(len(X_test))   # Prints the no. of samples in the testing set which is 20% of the training set
# model = SVC(kernel='linear')   # Creates a svm model. The parameters, ‘C’, ‘gamma’, ‘kernel’ can be varied as arguments
# print(model.fit(X_train, y_train)) # The model is trained at this point
# print(model.score(X_test, y_test))   # The accuracy of the model is calculated
