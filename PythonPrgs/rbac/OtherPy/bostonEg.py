import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# import Tkinter as tk
from tkinter import Tk, BOTH
from tkinter.ttk import Frame

# import tkinter


# import tkMessageBox
# # 1. Import the data
# from sklearn.datasets import load_boston
#
# # 2. Collect the dat
# boston=load_boston()
#     #print(boston)
#
# # 3. Clean the data
# df_x = pd.DataFrame(boston.data,
#                     columns=boston.feature_names)
# df_y = pd.DataFrame(boston.target)
# print( "Target : ", boston.target)
# print("df_y :------------------------ \n")
# print(df_y.keys())
# print("_____________________________________________________________-")
# print("Descriptive Statistics of the Boston dataset : ", df_x.describe())
# print("Printing Series...")
# print(df_x.columns.values)
# for item in df_x.columns.values:
#     print(item)
#
# print ("* * * * * * * * * * * * *")
# for col in df_x.columns:
#     series =df_x[col]
#
# print("Series : \n",series)
# # 4. Split the data
# x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)
#
# # 5. Create the model
# reg = linear_model.LinearRegression()
#
# # 6. Train the model
# reg.fit(x_train, y_train)
# print("Regression Coeff : ",reg.coef_)
# print("Regression Intercept : ",reg.intercept_)
#
# # 7. Make predictions
# a = reg.predict(x_test)
# print(f"Predicting the value of {a[5]}")
# print("y_test values : ",y_test)
# # Evaluate and Improve
# print("Mean : ", np.mean((a-y_test)**2))

root = Tk()
# tkinter._test()
# UserNameLab = Label(root, text = "Useraname")
# PwdLab = Label(root, text = "Password")
# UserNameLab.pack()

root.geometry("250x150+300+300")
root.mainloop()
