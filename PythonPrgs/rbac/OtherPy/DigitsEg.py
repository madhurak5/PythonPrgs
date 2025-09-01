import pandas as pd
import numpy as np
# import tensorflow
# from PIL import Image

# from keras.models import Sequential
# from keras.losses import categorical_crossentropy
from matplotlib import pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# The data is fetched
from sklearn.tests.test_random_projection import test_input_size_jl_min_dim

wine = load_wine()
print (dir (wine))

print("Feature Names : ", wine.feature_names) # Prints the column names from the data

print(wine.data) # Prints the data values
print("Individual column : ", wine['target_names'])

df = pd.DataFrame(wine.data, columns=wine.feature_names) # Creates the dataframe along with the column/feature names
print(df.head()) # Prints the first 5 rows from the dataframe
df['target'] =wine.target # Creates a new column, ‘target’, in the dataframe
print(df.head())

print(wine.target_names) # Prints the unique names in the column ‘target’
noOfAttr = len(wine.feature_names)
print("No. of Attributes : ",noOfAttr)
for i in range(0, noOfAttr):
    print(wine.feature_names[i])


print(df[df.target == 0])  # Prints the dataframe where the target value is 0
df['Wine Name'] = df.target.apply(lambda x:wine.target_names[x])  # Creates a new column ‘Flower Name’ and appends it to the dataframe. Lambda function helps to create or generate another column by applying it to the target column.

# print(df)
df0 = df[df.target == 0]   # Creates a new dataframe containing only those rows where the target is 0
df1 = df[df.target == 1] # Creates a new dataframe containing only those rows where the target is 1
df2 = df[df.target == 2] # Creates a new dataframe containing only those rows where the target is 2
print(df0.head())
print(df1.head())
print(df2.head())

plt.xlabel('alcohol')   # In the Scatter plot, the x-axis is labeled
plt.ylabel('malic_acid')   # In the Scatter plot, the y-axis is labeled
plt.scatter(df0['alcohol'], df0['malic_acid'], color = 'green', marker = '+')
plt.scatter(df1['alcohol'], df1['malic_acid'], color = 'red', marker = '+')
plt.scatter(df2['alcohol'], df2['malic_acid'], color = 'blue', marker = '+')
# plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color = 'blue', marker = '*')
# plt.scatter(df2['sepal length (cm)'], df2['sepal width (cm)'], color = 'red', marker = "+")
plt.show()
#
X = df.drop(['target', 'Wine Name'],axis='columns')  # Creates dataframe, X, by dropping the columns, ‘target’ and ‘Flower Name’

# print(X.head())
y = df.target
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # Splits the data into training and test datasets, 20% as testing set

print(len(X_train)) # Prints the no. of samples in the training set
print(len(X_test))   # Prints the no. of samples in the testing set which is 20% of the training set
model = SVC(kernel='linear')   # Creates a svm model. The parameters, ‘C’, ‘gamma’, ‘kernel’ can be varied as arguments
print(model.fit(X_train, y_train)) # The model is trained at this point
print(model.score(X_test, y_test))   # The accuracy of the model is calculated

