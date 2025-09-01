import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn import linear_model
# import keras

# Data - imported
wine = load_wine()
print(wine)


# Data - cleaned
df_x = pd.DataFrame(wine.data, columns=wine.feature_names)
df_y = pd.DataFrame(wine.target)
print(df_x.shape)
noOfAttr = len(df_x.keys())
attr = df_x.keys()
print("No. of attributes : ", noOfAttr)
print("Attributes are : ",attr)
for i in range(0, noOfAttr):
    print(attr[i])

print("Descriptive Statistics of the wine dataset : \n", df_x.describe())
print("Columns in the dataset : \n",df_x.columns.values)

# Data - split
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)

# model - created
reg = linear_model.LinearRegression()

# model - trained
reg.fit(x_train, y_train)
print("Regression Coeff : ",reg.coef_)
print("Regression Intercept : ",reg.intercept_)

#Predictions
pre_val = reg.predict(x_test)
print("Predicted value : ", pre_val)
print((pre_val[0]))
print("Mean : ", np.mean(pre_val-x_test)**2)

un = gtk.entry.get_text("unip")
