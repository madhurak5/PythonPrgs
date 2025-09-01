import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 200)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error

# Load the data
carData = pd.read_csv("D://PythonPrgs/csvFiles/used_cars_data.csv")
carData1 = carData.copy()

# Check shape of the data
print(f"There are {carData.shape[0]} rows and {carData.shape[1]} cols.")

# View 5 random rows of the data
np.random.seed(1)
carData.sample(n = 5)

# View top 5 rows of the data
print(carData.head())

# Columns in the data
cols = carData.columns
print(cols)

#Check datatypes
print(carData.info())

# Check duplicate value
print(carData.duplicated().sum())

# Check for missing values.
print(carData.isnull().sum())

# Check the statistical summary of the data
print(carData.describe().T)

#Check for categorical columns
catCols = carData.select_dtypes(exclude=np.number).columns
print(catCols)

# Get value counts for the non-numeric columns
for i in catCols:
    print(carData[i].value_counts())

#  Drop columns with woo many unique values and too many missing values
carData = carData.drop(['S.No.', 'Name', 'New_Price'], axis=1)
print(carData.shape)
cols = carData.columns
print(cols)

# Univariate Analysis
# plt.figure(figsize=(12, 8))
# for i in ['Location', 'Fuel_Type', 'Transmission', 'Owner_Type']:
#     sns.barplot(carData[i], carData['Price'])
    # plt.show()
# sns.barplot("Fuel_Type", "Price", data=carData)
# sns.pairplot(carData[['Kilometers_Driven','Price']])
# sns.countplot(x="Location",data=carData1)
# sns.countplot(x="Year",data=carData1)
# sns.countplot(x="Kilometers_Driven",data=carData1)
# sns.countplot(x="Owner_Type",data=carData1)
# sns.countplot(x="Seats",data=carData1)
# sns.countplot(x="Transmission",data=carData1)
# plt.show()
print(carData1[['Price', 'New_Price']].head(15))
# print("Quantile : ", np.quantile(carData['Mileage'], [0.25, 0.75]))
from sklearn.preprocessing import LabelEncoder
labEnc = LabelEncoder()

# sns.pairplot(data=carData,hue='Location')
# sns.barplot(x=carData['Location'],y=carData['Price'])
# sns.countplot('Location',hue='Transmission', data=carData)
# sns.countplot('Location',hue='Fuel_Type', data=carData)
# sns.countplot('Location',hue='Owner_Type', data=carData)
# sns.countplot('Fuel_Type',hue='Transmission', data=carData)
# sns.countplot('Fuel_Type',hue='Owner_Type', data=carData)
# sns.countplot('Transmission',hue='Owner_Type', data=carData)
# sns.lineplot(carData['Fuel_Type'], carData['Price'], hue=carData['Transmission'], ci=0)
# sns.lineplot(carData['Owner_Type'], carData['Price'], hue=carData['Transmission'], ci=0)
# sns.lineplot(carData['Location'], carData['Price'], hue=carData['Transmission'], ci=0)
# sns.lineplot(carData['Owner_Type'], carData['Price'], hue=carData['Fuel_Type'], ci=0)
# sns.lineplot(carData['Fuel_Type'], carData['Price'], hue=carData['Owner_Type'], ci=0)
# sns.lineplot(carData['Seats'], carData['Price'], hue=carData['Fuel_Type'], ci=0)
carData['Location'] = pd.get_dummies(carData['Location'])
# plt.show()
print("*"*60)
print(carData["Location"].head())
carData['Location'] = labEnc.fit_transform(carData['Location'])
# carData['Fuel_Type']
encCols = ['Location','Fuel_Type', 'Transmission', 'Owner_Type']
for i in encCols:
    carData[i] = labEnc.fit_transform(carData[i])


def repVal(col2):
    if isinstance(col2, str):
        no3 = col2.split(" ")
        return float(no3[0])

numCols = ['Mileage', 'Engine', 'Power']
for i in numCols:
    carData[i] = carData[i].apply(repVal)

for i in numCols:
    carData[i].fillna(value=carData[i].median(), inplace=True)
# print(carData.head())


for i in ['Seats', 'Price']:
    carData[i].fillna(value=carData[i].median(), inplace=True)
print(carData.head())
print(carData.dtypes)
# sns.distplot(carData['Mileage']) #, hue=carData['Fuel_Type'])
# sns.boxplot(x=carData1['Fuel_Type'], y=carData['Price'])

# sns.pairplot(carData)
# sns.countplot(carData['Location'], )
# distCols = ['Year','Kilometers_Driven','Mileage','Engine','Power','Seats', 'Price']
# for i in distCols:
#     sns.distplot(carData[i], kde=True) # sns.distplot(carData['Price'], kde=True)
    # plt.show()


# plt.figure(figsize=(10, 8))
# correl = carData.corr()
# sns.heatmap(correl, annot=True)
# plt.show()
quartile = np.quantile(carData['Year'][carData['Year'].notnull()],[0.25, 0.75])
power_4iqr = 4 * (quartile[1] - quartile[0])

print(f'Q1 = {quartile[0]}, Q3 = {quartile[1]}, 4 * IQR = {power_4iqr}')
outlier_power = carData.loc[np.abs(carData['Year'] - carData['Year'].median()) > power_4iqr, 'Year']
print("Outlier Power : ", outlier_power)

quartile = np.quantile(carData['Mileage'][carData['Mileage'].notnull()],[0.25, 0.75])
power_4iqr = 4 * (quartile[1] - quartile[0])
print(f'Q1 = {quartile[0]}, Q3 = {quartile[1]}, 4 * IQR = {power_4iqr}')
outlier_power = carData.loc[np.abs(carData['Mileage'] - carData['Mileage'].median()) > power_4iqr, 'Mileage']
print("Outlier Power Mileage: ", outlier_power)

quartile = np.quantile(carData['Kilometers_Driven'][carData['Kilometers_Driven'].notnull()],[0.25, 0.75])
power_4iqr = 4 * (quartile[1] - quartile[0])
print(f'Q1 = {quartile[0]}, Q3 = {quartile[1]}, 4 * IQR = {power_4iqr}')
outlier_power = carData.loc[np.abs(carData['Kilometers_Driven'] - carData['Kilometers_Driven'].median()) > power_4iqr, 'Kilometers_Driven']
print("Outlier Power Kilometers_Driven: ", outlier_power)

carData['Kilometers_Driven'].hist(bins=20)
plt.title('Power before exaggerating the outliers')
plt.show()
print(carData['Kilometers_Driven'].mean())
carData.loc[outlier_power.index, 'Kilometers_Driven'] = [-200000.0, -200000.0, -200000.0, -200000.0,-200000.0, -200000.0, -200000.0, -200000.0,-200000.0, -200000.0, -200000.0, -200000.0,-200000.0, -200000.0, -200000.0, -200000.0,-200000.0, -200000.0, -200000.0, -200000.0,-200000.0, -200000.0, -200000.0, -200000.0,-200000.0, -200000.0, -200000.0, -200000.0,-200000.0, -200000.0, -200000.0]
carData['Kilometers_Driven'].hist(bins=20)
plt.title('Power after exaggerating outliers')
plt.show()


quartile = np.quantile(carData['Power'][carData['Power'].notnull()],[0.25, 0.75])
power_4iqr = 4 * (quartile[1] - quartile[0])
print(f'Q1 = {quartile[0]}, Q3 = {quartile[1]}, 4 * IQR = {power_4iqr}')
outlier_power = carData.loc[np.abs(carData['Power'] - carData['Power'].median()) > power_4iqr, 'Power']
print("Outlier Power Power: ", outlier_power)

quartile = np.quantile(carData['Price'][carData['Price'].notnull()],[0.25, 0.75])
power_4iqr = 4 * (quartile[1] - quartile[0])
print(f'Q1 = {quartile[0]}, Q3 = {quartile[1]}, 4 * IQR = {power_4iqr}')
outlier_power = carData.loc[np.abs(carData['Price'] - carData['Price'].median()) > power_4iqr, 'Price']
print("Outlier Power Price: ", outlier_power)


X = carData
y = carData['Price']
print(X.columns)
#
# # Split the data into Training and Testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
linReg = LinearRegression()
linReg.fit(X_train, y_train)
coef_df = pd.DataFrame(
    np.append(linReg.coef_, linReg.intercept_),
    index=X_train.columns.tolist() + ["Intercept"],
    columns=["Coefficients"],
)
print(coef_df)
# print("#"*25, "Testing Data perf", "#"*25)
# linReg.fit(X_test, y_test)
# coef_df = pd.DataFrame(
#     np.append(linReg.coef_, linReg.intercept_),
#     index=X_test.columns.tolist() + ["Intercept"],
#     columns=["Coefficients"],
# )
# print(coef_df)
#
# # function to compute adjusted R-squared
# def adj_r2_score(predictors, targets, predictions):
#     r2 = r2_score(targets, predictions)
#     n = predictors.shape[0]
#     k = predictors.shape[1]
#     return 1 - ((1 - r2) * (n - 1) / (n - k - 1))
#
#
# # function to compute MAPE
# def mape_score(targets, predictions):
#     return np.mean(np.abs(targets - predictions) / targets) * 100
#
# # function to compute different metrics to check performance of a regression model
# def model_performance_regression(linRegModel, X_train, y_train):
#     """
#     Function to compute different metrics to check regression model performance
#
#     model: regressor
#     predictors: independent variables
#     target: dependent variable
#     """
#
#     # predicting using the independent variables
#     pred = linReg.predict(X_train)
#
#     r2 = r2_score(y_train, pred)  # to compute R-squared
#     adjr2 = adj_r2_score(X_train, y_train, pred)  # to compute adjusted R-squared
#     rmse = np.sqrt(mean_squared_error(y_train, pred))  # to compute RMSE
#     mae = mean_absolute_error(y_train, pred)  # to compute MAE
#     mape = mape_score(y_train, pred)  # to compute MAPE
#
#     # creating a dataframe of metrics
#     carData_perf = pd.DataFrame(
#         {
#             "RMSE": rmse,
#             "MAE": mae,
#             "R-squared": r2,
#             "Adj. R-squared": adjr2,
#             "MAPE": mape,
#         },
#         index=[0],
#     )
#
#     return carData_perf
# print("*"*100)
# print("Training Performance")
# print("-"*50)
# linReg_performance = model_performance_regression(linReg, X_train, y_train)
# print("Performance of Linear Regression:")
# print(linReg_performance)
#
# print("*"*100)
# print("Testing Performance")
# print("-"*100)
# linReg_performance = model_performance_regression(linReg, X_test, y_test)
# print("Performance of Linear Regression on Testing Data:")
# print(linReg_performance)
# print("*"*100)
#
# pred = linReg.predict(X_test) # X_test
#
# y_pred = linReg.predict(X_train) # X_test
# r2score = r2_score(y_train, y_pred) # y_test
# print("r2_score : ", r2score)
# n = X_test.shape[0]
# k = X_test.shape[1]
# adjr2 = 1 - ((1 - r2score) * (n - 1) / (n - k - 1))
# print("Adjusted r2 score : ", adjr2)
# rmse = np.sqrt(mean_squared_error(y_train, y_pred)) # y_test
# print("RMSE : ", rmse)
# mae = mean_absolute_error(y_train, y_pred) # y_test
# print("Mean Absolute Error : ", mae)
