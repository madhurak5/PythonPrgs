import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# adult = pd.read_csv("D://PythonPrgs/csvFiles/adult_data.csv")
# test = pd.read_csv("D://PythonPrgs/csvFiles/adult_test.csv")
# print("Data: \n",adult.head(), adult.shape)
# print(adult.dtypes)# nunique()
# print(adult['education'].value_counts())
# # print(np.average(adult['hours-per-week']))
# print(adult['hours-per-week'].mean())
# sns.set({'figure.figsize': (8,7)})
# sns.countplot(x=adult['salary'],hue=adult['sex'])
# plt.show()
# print("Test: \n",test.head(), test.shape)
# final_df = pd.concat([adult, test], join='inner')
# print(final_df.shape)
# print("Average value of hours-per-week : ", np.average(final_df['hours-per-week']))
# country = np.mean(adult['hours-per-week'])
# print("Mean: ", country)
# count1 = pd.Series(adult.groupby(['native-country'])['hours-per-week'].mean())
# print(count1.sort_values(ascending=False))
# # What is the average value of "capital-gain" for the dataset "adult_data.csv"?
# avg_cap_gain = np.average(adult['capital-gain'])
# print(avg_cap_gain)
# # sns.distplot(adult['age'] , kde=True)
# # plt.show()
# print(adult[(adult['capital-loss'] >0)]) #len(data[(data['capital-loss'] > 0)])
# # hours-per-week
# # Which Occupation has the 2nd highest average working hours for dataset "adult_data.csv"?
# # occ1 = adult.groupby(['occupation'])[]
# # avg_occ =
# # print(pd.Series(adult.groupby(['occupation'])['hours-per-week']))
# count2 = pd.Series(adult.groupby(['occupation'])['hours-per-week'].mean())
# print(count2.sort_values(ascending=False))

# ----------------------------------------------------------------------------------------------------------------
# ch =1
# cardio = pd.read_csv("D://PythonPrgs/csvFiles/CardioGoodFitness.csv")
# print(cardio.head())
# def univariate_analysis(ft):
#     sns.barplot(cardio['Gender'], ft)
#     plt.show()
#
# uniFeat = input("Enter the input variable name for univariate analysis: ")
# univariate_analysis(cardio[uniFeat])
# ----------------------------------------------------------------------------------------------------------------
auto = pd.read_csv("D://PythonPrgs/csvFiles/autompg.csv")
print(auto.head())
print(auto.shape)
print(auto.describe())
hpIsDigit = pd.DataFrame(auto.horsepower.str.isdigit())
print()
auto[hpIsDigit['horsepower'] == False]
print(auto['horsepower'] == False)

