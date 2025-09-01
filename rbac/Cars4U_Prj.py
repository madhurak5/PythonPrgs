import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", 200)
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# carData = pd.read_csv("D://PythonPrgs/csvFiles/used_cars_data.csv")
# print(f'There are {carData.shape[0]} rows and {carData.shape[1]} cols')
# print(carData.head())
# print(carData.info())
# print(carData.describe().T)
# # print(carData.isnull().sum())
# cols = carData.columns
# # for i in cols:
# #     print(carData[i].isnull().sum())
# carData1 = carData.copy()
# # print(carData1.head())
# print(carData1['Mileage'].replace("km/kg","").replace("kmpl", ""))
# # for i in carData1['Mileage'].values:
# i = carData1['Mileage'][2].split(" ")
# print("Split data " , i[0])
# for i in range(0, len(carData1['Mileage'].values) - 1):
#     if isinstance(carData1['Mileage'][i], str):
#         spl = carData1['Mileage'][i].split(" ")
#         carData1['Mileage'][i] = spl[0]
#         # print("Split data ", spl[0])
# carData1['Mileage'] = pd.to_numeric(carData1['Mileage'],errors='coerce')
# print(carData1['Mileage'].dtype)
# for i in range(0, len(carData1['Engine'].values) - 1):
#     if isinstance(carData1['Engine'][i], str):
#         spl = carData1['Engine'][i].split(" ")
#         carData1['Engine'][i] = spl[0]
# carData1['Engine'] = pd.to_numeric(carData1['Engine'],errors='coerce')
# for i in range(0, len(carData1['Power'].values) - 1):
#     if isinstance(carData1['Power'][i], str):
#         spl = carData1['Power'][i].split(" ")
#         carData1['Power'][i] = spl[0]
# carData1['Power'] = pd.to_numeric(carData1['Power'],errors='coerce')
#
# for i in range(0, len(carData1['New_Price'].values) - 1):
#     if isinstance(carData1['New_Price'][i], str):
#         spl = carData1['New_Price'][i].split(" ")
#         carData1['New_Price'][i] = spl[0]
#     # if carData1['New_Price'][i].isNan():
#     #     carData1['New_Price'] = 0
#
# # carData1['New_Price'] = carData1['New_Price'].astype(str).replace('nan', 'is_missing').astype('category')
# print(carData1.head())
# print(carData1.info())
# print(carData1['New_Price'].value_counts())
# # print(carData1['Location'].value_counts().sort_values(ascending=False))
# # print(carData1['Year'].value_counts().sort_values(ascending=False))
# # print(carData1['Kilometers_Driven'].value_counts().sort_values(ascending=False))
# # print(carData1['Fuel_Type'].value_counts().sort_values(ascending=False))
# # print(carData1['Transmission'].value_counts().sort_values(ascending=False))
# # print(carData1['Owner_Type'].value_counts().sort_values(ascending=False))
# # print(carData1['Engine'].value_counts().sort_values(ascending=False))
# # print(carData1['Seats'].value_counts().sort_values(ascending=False))
# catCols = carData1.select_dtypes(include=np.number).columns
# print(catCols)
# # sns.barplot(carData1['Location'], carData1['Price'])
#
# # sns.heatmap(carData1.corr(), annot=True)
# # sns.scatterplot(carData1['Owner_Type'], carData1['Price']) #, hue=carData1['Engine'])
# # plt.show()
# # carData1['Power'] = carData1['Power'].astype(int)
# # print(carData1['Power'].dtype)
# print(carData1['Power'].sample(50))
#
# print(carData1['Power'].isnull().sum())
# print(carData1.isnull().sum(axis=1).value_counts())
# num_missing = carData1.isnull().sum(axis=1)
# # print(carData1[num_missing == 4].sample(10))
# # print(pd.get_dummies(carData1['Transmission'],drop_first=False).iloc[:10, :])
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# labEnc= LabelEncoder()
# carData1['Transmission'] = labEnc.fit_transform(carData1[['Transmission']])
# print(f"There are {carData1['Location'].isnull().sum()} missing values")
# carData1['Location'] = labEnc.fit_transform(carData1['Location'])
#
# print(f"There are {carData1['Year'].isnull().sum()} missing values")
# print(carData1['Year'].value_counts())
# print(carData1['Kilometers_Driven'].dtypes)
# print(carData1['Fuel_Type'].value_counts())
# carData1['Fuel_Type'] = labEnc.fit_transform(carData1['Fuel_Type'])
# print(carData1['Owner_Type'].dtypes)
# print(carData1['Owner_Type'].value_counts())
# carData1['Owner_Type'] = labEnc.fit_transform(carData1['Owner_Type'])
# print(carData1[['Transmission', 'Location', 'Fuel_Type', 'Owner_Type']].head())
# # print(carData1[['Mileage','Engine' ,'Power', 'Seats']].dtypes)
# # carData1['Seats'] = carData1['Seats'].astype(int)
# # print(carData1[['Mileage','Engine' ,'Power', 'Seats']].dtypes)
# print(f"There are {carData1['Seats'].isnull().sum()} missing values")
# # print(carData1['Seats'].median())
# carData1['Seats'].fillna(value=carData1['Seats'].median(), inplace=True)
# print(f"There are {carData1['Mileage'].isnull().sum()} missing values")
# print(carData1['Mileage'].median())
# carData1['Mileage'].fillna(value=carData1['Mileage'].median(), inplace=True)
# print(f"There are {carData1['Engine'].isnull().sum()} missing values")
# carData1['Engine'].fillna(value=carData1['Engine'].median(), inplace=True)
# print(f"There are {carData1['Power'].isnull().sum()} missing values")
# print(carData1['Power'].median())
# print(carData1['Power'].mean())
# carData1['Power'].fillna(value=carData1['Power'].median(), inplace=True)
# # print(f"There are {carData1['Power'].isnull().sum()} missing values")
# # print(carData1.head())
# print(f"There are {carData1['New_Price'].isnull().sum()} missing values")
# carData1['New_Price'] = carData1['New_Price'].astype(str).replace('nan', 'is_missing').astype('category')
# carData1 = carData1.drop(['New_Price'], axis =1)
# # print(carData1['New_Price'].median())
# # print(carData1['New_Price'].mean())
# # sns.scatterplot()
# carData1['Price'].fillna(value=carData1['Price'].mean(), inplace=True)
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# from mlxtend.feature_selection import SequentialFeatureSelector as SFS
#
#
# print(f"There are {carData1['Name'].isnull().sum()} missing values")
# carData1['Name'] = labEnc.fit_transform(carData1[['Name']])
# # print(carData1['Name'].value_counts())
# print(carData1.head())
# print(carData1.shape)
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# scaler = MinMaxScaler()
# carScale = scaler.fit_transform(carData1)
# X = carData1.drop(['Price'], axis=1)
# y = carData1['Price']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# from sklearn.linear_model import LinearRegression
# linReg = LinearRegression()
# print(linReg.fit(X_train, y_train))
# # print(carData1.isnull().sum())
# # print(carData1['Price'].median())
# # print(carData1['Price'].mean())
# coef_df = pd.DataFrame(np.append(linReg.coef_, linReg.intercept_),index=X_train.columns.tolist() + ["Intercept"],columns=["Coefficients"],)
# print("Coefficients :\n", coef_df)
# from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
# # kickData['scaledGoal'] = mScaler.fit_transform(kickData['usd_goal_real'].values.reshape(-1, 1))
# y_pred = linReg.predict(X_test)
# r2score = r2_score(y_test, y_pred)
# print("r2_score : ", r2score)
# n = X_test.shape[0]
# k = X_test.shape[1]
# adjr2 = 1 - ((1 - r2score) * (n - 1) / (n - k - 1))
# print("Adjusted r2 score : ", adjr2)
# print("r2_score : ", r2_score(y_test, y_pred))
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print("RMSE : ", rmse)
# mae = mean_absolute_error(y_test, y_pred)
# print("Mean Absolute Error : ", mae)
# sfs = SFS(linReg, k_features=X_train.shape[1],forward=True,floating=False,scoring='r2', cv=10)
# sfs1 = sfs.fit(X_train, y_train)
# print(sfs1)
# from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
# feat_cols = list(sfs.k_feature_idx_)
# print(feat_cols)
# fig1 = plot_sfs(sfs.get_metric_dict(), kind="std_err", figsize=(15, 5))
# plt.title("Sequential Forward Selection (w. StdErr)")
# plt.xticks(rotation=90)
# plt.show()
# pd.get_dummies(carData,drop_first=True)
# print("-----------------------------------------X_test----------------------------------------------")
# print(linReg.fit(X_test, y_test))
# # print(carData1.isnull().sum())
# # print(carData1['Price'].median())
# # print(carData1['Price'].mean())
# coef_df = pd.DataFrame(np.append(linReg.coef_, linReg.intercept_),index=X_test.columns.tolist() + ["Intercept"],columns=["Coefficients"],)
# print("Coefficients X_test:\n", coef_df)
# from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
# # kickData['scaledGoal'] = mScaler.fit_transform(kickData['usd_goal_real'].values.reshape(-1, 1))
# y_pred = linReg.predict(X_test)
# r2score = r2_score(y_test, y_pred)
# print("r2_score : ", r2score)
# n = X_test.shape[0]
# k = X_test.shape[1]
# adjr2 = 1 - ((1 - r2score) * (n - 1) / (n - k - 1))
# print("Adjusted r2 score : ", adjr2)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print("RMSE : ", rmse)
# mae = mean_absolute_error(y_test, y_pred)
# print("Mean Absolute Error : ", mae)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
insur = pd.read_csv("D://PythonPrgs/csvFiles/insurance.csv")
print(insur.head())
print(insur.shape)
print(insur['charges'].isnull().sum())
sns.distplot(insur['charges'])
plt.show()
print(insur['children'].value_counts(normalize=True))
insur['logbmi'] = np.log2(insur['bmi'])
sns.distplot(insur['bmi'], kde=True)
plt.show()
print(insur.groupby('region')['charges'].max())
print(insur.corr())
print(insur['region'].isnull().sum())
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
ohe = LabelEncoder()
insur['sex'] = ohe.fit_transform(insur['sex'])
insur['smoker'] = ohe.fit_transform(insur['smoker'])
insur = pd.get_dummies(insur,drop_first=True)
print(insur.head())
X = insur.drop('charges', axis=1)
print(X.columns)
print(X.dtypes)
y = insur['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
linReg = LinearRegression()
linReg.fit(X_train, y_train)
coef_df = pd.DataFrame(
    np.append(linReg.coef_, linReg.intercept_),
    index=X_train.columns.tolist() + ["Intercept"],
    columns=["Coefficients"],
)
print(coef_df)
pred = linReg.predict(X_test)
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
y_pred = linReg.predict(X_train) # X_test
r2score = r2_score(y_train, y_pred) # y_test
print("r2_score : ", r2score)
n = X_test.shape[0]
k = X_test.shape[1]
adjr2 = 1 - ((1 - r2score) * (n - 1) / (n - k - 1))
print("Adjusted r2 score : ", adjr2)
rmse = np.sqrt(mean_squared_error(y_train, y_pred)) # y_test
print("RMSE : ", rmse)
mae = mean_absolute_error(y_train, y_pred) # y_test
print("Mean Absolute Error : ", mae)
