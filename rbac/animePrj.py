import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 200)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# # ------------------------------------------------------------------------------------------------------------------------
# anime_df = pd.read_csv("D://PythonPrgs/csvFiles/anime_data.csv")
# print("Shape of data: ", anime_df.shape)
# print(anime_df.head())
# animeData = anime_df.copy()
# print("There are ", animeData.duplicated().sum(), " duplicate values in the data")
# print("Columns in the data are : \n", animeData.columns)
# print("Information about the data : ")
# print(animeData.info())
# print("Missing values info:")
# print(animeData.isnull().sum())
# print("Statistical summary of the data :")
# print(animeData.describe().T)
# catCols = animeData.select_dtypes(exclude=np.number).columns
# print("Categorical Columns are : \n", catCols)
# catCols1 = ['mediaType', 'ongoing', 'sznOfRelease','studio_primary']
# print(catCols1)
# for i in catCols1:
#     print(animeData[i].value_counts())
#     print("-" * 50)
# animeData.drop(['title', 'description'], axis=1, inplace=True)
# animeData.dropna(inplace=True)
# print(f'There are {animeData.shape[0]} rows and {animeData.shape[1]} columns')
# X = "DataScience"
# print(X.endswith("e"))
# Country = 'United-States-Of-America'
# print(Country.split('-')[0])
# data = pd.DataFrame({"Col1": [100,200,300,400], "Col2":[500,600,700,800], "Col3":[900,1000,1100,1200], "Col4":["Nature","Wildlife","Animals","Humans"]})
# print(data)
# data["Col5"] = data[["Col1","Col2","Col3"]].sum(axis=1)
# print(data)
# data["Col6"] = data["Col3"] - data["Col1"]
# print(data)
# # data.drop(["Col2"])
# # print(data)
# data['Col1'] = data['Col1'].apply(lambda x : x*5)
# print(data)
# print(data['Col4'].str.endswith('s').sum())
# print(data["Col4"].str.upper())
# df = pd.DataFrame({'A': ['m', 'f', 'm'], 'B': ['b', 'a', 'c'],'Age': [11, 12, 13]})
# df = pd.get_dummies(df, prefix=['Gender', 'Class'])
#
# t1 = [2, 5, 12, 15, 19, 4, 6, 11, 16, 18, 12, 12, 42, 6, 56, 34, 23, 11]
# t2 = t1.sort_values()
# print(t2)

# ------------------------------------------------------------------------------------------------------------------------

kickData = pd.read_csv("D://PythonPrgs/csvFiles/KickStarterProjects.csv")
print(kickData.shape)
print(kickData.describe().T)
print(kickData.info())
kickData = kickData.drop(['currency', 'goal'], axis=1)

print(kickData.shape)
print(kickData.isnull().sum())
kickData = kickData.dropna()
print(kickData.shape)
print(kickData.isnull().sum())
stateUni = kickData['state'].nunique()
print(stateUni)
print(f'there are {stateUni} unique projects and ')
print(kickData['state'].value_counts())
print(kickData.head())
sns.distplot(kickData['usd_goal_real'])
plt.show()
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# mScaler = MinMaxScaler()
mScaler = StandardScaler()
kickData['scaledGoal'] = mScaler.fit_transform(kickData['usd_goal_real'].values.reshape(-1, 1))
# df['Col1_scaled'] = scaler.fit_transform(df['Col1'].values.reshape(-1,1))
print(kickData.head())
# sns.distplot(kickData['scaledGoal'])
# plt.show()
kickData['usd_goal_real'] = np.log(kickData['usd_goal_real'])
# sns.distplot(kickData['usd_goal_real'])
# plt.show()
print(kickData['category'].value_counts())
dupProj = kickData['name'].duplicated().any() # True
print(dupProj)