import pandas as pd
import  numpy as np
import pandas_profiling
from sklearn.preprocessing import LabelEncoder
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz, DecisionTreeClassifier
import pydotplus
from sklearn.linear_model import LogisticRegression
from  sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn import  datasets
# from sklearn.datasets import make_classification
from sklearn import svm
# data = pd.read_csv("D://PythonPrgs/csvFiles/userprofile.csv")
# noOfCols = len(data.columns)
# print(noOfCols)
# import sys
# print("Profile Report .........................")
# pandas_profiling.ProfileReport(data).to_file("profile_output.html")
# # sns.heatmap(data.isna(),yticklabels=False, cbar=False, cmap='viridis')
# sns.distplot(data['activity'].dropna(),kde=False, color='darkred',bins=10)
# plt.show()
# print(data.head())
# for i in range(0, len(data['activity'])):
#     if data.loc[i, 'activity'] in ['professional', 'working-class']:
#         data.loc[i, 'class'] = "owner"
#     else:
#         data.loc[i, 'class'] = "consumer"
# X = data.iloc[:,0:18]
# y = data.iloc[:,19]
# lab_x = LabelEncoder()
# X = X.apply(LabelEncoder().fit_transform)
#
# x_train, x_test,y_train,  y_test = train_test_split(X,y, test_size=0.7, random_state=1)
# # print(X.head())
# # # print(y)
#
# print(X)
# print(y)
# regr = DecisionTreeClassifier()
# regr.fit(x_train, y_train)
# print(regr)
# # # X_in = np.array([2,27, 50])
# # # X_in = np.array([3,3,94])
# y_pred = regr.predict(x_test)
# print("Predictions:\n",y_pred)
# cm = confusion_matrix(y_test,y_pred)
# print("Confusion Matrix :\n", cm)
# print("Accuracy of DT : ", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# # dot_data = StringIO()
# # export_graphviz(regr, out_file=dot_data, filled=True, rounded=True, special_characters=True)
# # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# # graph.write_png('treeDT.png')
# #
#
# # # x,y = make_classification(n_samples=1000, n_features=25, n_classes=2, n_clusters_per_class=1, n_redundant=0, n_repeated=False)
# # # print(x)
# # # print(y)
# #
# # # print(x_train.shape)
# # # print(y_train.shape)
#
#
# log_reg = LogisticRegression()
# print(log_reg.fit(x_train, y_train))
# # X_in = np.array([3,3,94])
# # # X_in = np.array([2,27, 50])
# y_pred = log_reg.predict(x_test)
# cm = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix LR : \n", cm)
# print("Accuracy of LogReg : ", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# x_train, x_test,y_train,  y_test = train_test_split(X,y, test_size=0.7, random_state=1)
# clss = SVC(kernel='rbf')
# clss.fit(x_train,y_train)
# y_pred = clss.predict(x_test)
# print("A Score of svc : ", accuracy_score(y_test, y_pred))
#
# fig = px.scatter_3d(X, x = 'activity', y = 'latitude', z = 'weight')
# fig.show()
#
#
#
# cls = svm.SVC(gamma=0.001,C=100)
# print("SVM classifier: \n", cls)
# cls.fit(x_train, y_train)
# pred = cls.predict(x_test)
# cms = confusion_matrix(y_test,pred)
# print(cms)
# print(classification_report(y_test, pred))
# print("Accuracy of svm: ", accuracy_score(y_test, pred))

# # canData = datasets.load_breast_cancer()
# digits = datasets.load_digits()
# clf = svm.SVC(gamma=0.001, C=100)
# X, y = digits.data[:-10], digits.target[:-10]
# clf.fit(X,y)
# print(clf.predict(digits.data[:-10]))
# plt.imshow(digits.images[0], interpolation='nearest')
# plt.show()
# X_train, X_test, y_train, y_test = train_test_split(canData.data, canData.target, test_size=0.3, random_state=1)
# cls = svm.SVC(kernel='linear')
# cls.fit(X_train, y_train) # train the model
# pred = cls.predict(X_test)
# print(classification_report(y_test, pred))
#
# x = np.linspace(-5.0, 5.0, 100)
# y = np.sqrt(10**2 - x ** 2)
# y = np.hstack([y,-y])
# x = np.hstack([x, -x])
# x1 = np.linspace(-5.0, 5.0, 100)
# y1 = np.sqrt(5**2 - x1 ** 2)
# y1 = np.hstack([y1,-y1])
# x1 = np.hstack([x1, -x1])
# plt.scatter(y, x)
# plt.scatter(y1, x1)
# plt.show()
# df1 = pd.DataFrame(np.vstack([y,x]).T, columns=['X1', 'X2'])
# df1['Y'] = 0
# df2 = pd.DataFrame(np.vstack([y1,x1]).T, columns=['X1', 'X2'])
# df2['Y'] = 1
# df = df1.append(df2)
# print(df.head(3))
# X = df.iloc[:, :2]
# y = df.Y
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# classifier = SVC(kernel='linear')
# print(classifier.fit(X_train, y_train))
# y_pred = classifier.predict(X_test)
# print("Accuracy score of SVC : ", accuracy_score(y_test, y_pred))
# df['X1Sq'] = df['X1'] ** 2
# df['X2Sq'] = df['X2'] ** 2
# df['X1*X2'] = df['X1'] * df['X2']
# print(df.head())
# X = df[['X1','X2', 'X1Sq', 'X2Sq', 'X1*X2']]
# y = df['Y']
# print(y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
# print(X_train)
# import plotly.express as px
# fig = px.scatter_3d(df, x = 'X1', y = 'X2', z = 'X1*X2', color='Y')
# fig.show()
# classifier = SVC(kernel='poly')
# print(classifier.fit(X_train, y_train))
# y_pred = classifier.predict(X_test)
# print("Accuracy score of SVC1 : ", accuracy_score(y_test, y_pred))
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
# print(X_train)
# import plotly.express as px
# fig = px.scatter_3d(df, x = 'X1Sq', y = 'X2Sq', z = 'X1*X2', color='Y')
# fig.show()

# --------------------------------------------------------------------------------------------------------
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# #
# tips_data = pd.read_csv("D://PythonPrgs/csvFiles/CardioGoodFitness.csv")
# # print(tips_data.info())
# # print(tips_data.shape)
# print((tips_data.head(5)))
# sns.distplot(tips_data['total_bill'], kde=False, rug=True) #, color="hotpink")
# sns.boxplot(y=tips_data["total_bill"], data=tips_data)

# sns.distplot(tips_data['tip'], kde=True, rug=True, color="lime")
# sns.boxplot(y=tips_data["tip"], data=tips_data, orient='v')

# sns.pointplot(tips_data["tip"], tips_data["total_bill"])

# sns.jointplot(tips_data["tip"], tips_data["total_bill"]) #, kind="hex", color="red")
# sns.pairplot(tips_data[['tip', 'total_bill']])
# sns.pairplot(tips_data[['total_bill','tip']])
# sns.lmplot(y="tip",x="total_bill", data=tips_data)
# sns.stripplot(tips_data["tip"], tips_data["sex"])
# sns.stripplot(tips_data["sex"], tips_data["tip"])
# sns.boxplot(tips_data['sex'], tips_data['tip'])
# sns.lmplot(y="tip",x="total_bill", data=tips_data,fit_reg=False,hue="sex")

# sns.stripplot(x=tips_data["day"],y=tips_data["total_bill"], hue=tips_data["day"], jitter=False)
# sns.swarmplot(tips_data["day"],tips_data["total_bill"], hue=tips_data["day"])
# sns.boxplot(x=tips_data["day"],y=tips_data["total_bill"], hue=tips_data["day"])
# sns.countplot(tips_data["day"], hue=tips_data["sex"])
# sns.lmplot(tips_data["total_bill"], tips_data["tip"], hue=tips_data["day"],row="sex",col="time")
# sns.lmplot(x='total_bill',y='tip',data=tips_data,fit_reg=False,hue='day',row='sex',col='time')
# sns.stripplot(x=tips_data["smoker"], y=tips_data["tip"])
# sns.boxplot(x=tips_data["smoker"], y=tips_data["tip"])
# sns.countplot(tips_data["smoker"])
# sns.lmplot(x='total_bill',y='tip',data=tips_data,fit_reg=False,hue='sex',col='smoker',palette='rocket')
# plt.bar(tips_data["sex"], tips_data["total_bill"])
# sns.barplot(tips_data["sex"],tips_data["total_bill"])
# sns.jointplot(tips_data["tip"], tips_data["total_bill"])

# plt.show()

# --------------------------------------------------------------------------------------------------------
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# tips_data = pd.read_csv("D://PythonPrgs/csvFiles/userprofile.csv") #KDDTestNew117
# print(tips_data.info())
# print(tips_data.isnull().any().any())
# print((tips_data.head(5)))
# # sns.jointplot(tips_data['dst_bytes'],tips_data['src_bytes']) #, hue=tips_data['protocol_type'])
#
# sns.swarmplot(tips_data['protocol_type'], tips_data['src_bytes'], hue=tips_data['protocol_type'])
# sns.barplot(tips_data['protocol_type'], tips_data['src_bytes'], hue=tips_data['protocol_type'])
# plt.show()
# --------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#
# fit_data = pd.read_csv("D://PythonPrgs/csvFiles/CardioGoodFitness.csv")
# # Understanding the structure of the data
# pd.set_option('display.max_columns',None)
# print("Dataset Shape: ", fit_data.shape) # Given dataset contains 180 rows and 9 columns
# print("Datatypes of various columns:\n", fit_data.dtypes) # There are 6 numerical variables (Age, Education, Usage, Fitness, Income, Miles) and 3 categorical variables (Product, Gender, MaritalStatus)
# print(fit_data.info()) # There are 6 numerical variables (Age, Education, Usage, Fitness, Income, Miles) and 3 categorical variables (Product, Gender, MaritalStatus)
# print("Statistical description of the dataset:\n",fit_data.describe()) # describes the statistical characteristics of the dataset for the numerical columns. # It indicates that the median age of the customers using the fitness products is 26
# print("Missing values information:\n", fit_data.isnull().sum()) # Shows that none of the columns contain any missing values

# Univariate Data Analysis
# sns.distplot(fit_data['Age']) # Most of the customers are in the age range of 25 to 30 with the median age being 26
# sns.distplot(fit_data['Education']) # Customers have an education of 12 to 21 years with the median years of education being 16
# sns.distplot(fit_data['Usage']) # Customers use the products 2 to 7 times a week, with using 3 times a week topping the usage list
# sns.distplot(fit_data['Fitness']) # Customer's self rated fitness level ranges from 1 to 5. Maximum number of customers have self rated themselves with a fitness level of 3, with the next set of customers rating themselves with a level of 5. Number of customers rating themselves with a 2 or 4 is the approximately same.
# sns.distplot(fit_data['Income']) # The income of the customers ranges from 30000 to 100000 approximately with most customers earning an income of 50000
# sns.distplot(fit_data['Miles']) # Number of miles that the customers run ranges from 10 to 350 miles, with maximum number of customers running 90 miles approximately
# print(fit_data['Age'].min())
# sns.boxplot(fit_data['Age']) # Age ranges from 18 - 50, with the median age being 26.
# sns.boxplot(fit_data['Education']) # Most customers have an education of 14 - 16 years, with min being 12 and max being 18. There are ver few customers with an education of 20 and 24 years of education that are the outliers in the current problem
# sns.boxplot(fit_data['Usage']) # Treadmills are used 2 - 5 times a week, with most usage being 3 and 4 times a week, SOme outliers show that the treadmills are user 6 and 7 times a week.
# sns.boxplot(fit_data['Fitness']) # Fitness level ranges from 2 to 5 with max no. of customers rating themselves with 3 and 4 as their fitness level. There is one outlier showing the level of fitness as 1
# sns.boxplot(fit_data['Miles']) # Customers rum from 20 miles to 175 miles with median miles being 95. Most customers rum from 60 - 115 miles
# sns.boxplot(fit_data['Income'])  # Customers' income ranges from 30000 to 70000 to 78000, with median income being 50000 and most cusomters have an income from 45000 to 59000
# sns.boxplot(x="Product", y="Age", data=fit_data, hue="Gender") # No of male customers using the 3 products is more than that of female customers. Even their age range is broader than that of female customers.
# sns.boxplot(x="Product", y="Age", data=fit_data, hue="Fitness") # Customers with fitness level 5 use TM798, with 4 use all 3 products, with 3 use all 3 products though they mostly use TM195 and TM498, with 2 use only TM195 and TM498
# sns.boxplot(x="Product", y="Age", data=fit_data, hue="MaritalStatus") # Partnered customers start at a later age compared to the single customers and go up to 50 and 48 years of age for TM195 and TM498 respectively, whereas for TM798, the max age is 38


# sns.countplot(fit_data['Age']) # Max no. of customers (25) are 25 years of age, followed by 18 who are 23 years of age and 12 customers each of 25 and 26 years of age.
# sns.countplot(fit_data['Education']) # 85 customers have 16  years of education followed by 55 customers who have 14 years of education and then 21 customers who have 18 years of education
# sns.countplot(fit_data['Usage']) # 69 customers use the treadmill 3 times a week, 51 - 4 times a week, 31 - 2 times a week, 19 - 5 times a week
# sns.countplot(fit_data['Fitness']) # 90 customers have self rated thier fitness level as 3, 30 as 5, 26 as 2 and 25 as 4. Only 1 customer has self rated his/her fitness level as 1
# sns.countplot(fit_data['Miles']) # 30 customers run for 85 miles, 12 for 95 miles, 10 each for 66 and 75 miles, and so on
# sns.countplot(fit_data['Income']) # A max of 14 customers have an income of 50000

# sns.pairplot(fit_data,hue="Product")
# sns.pairplot(fit_data[["Age", 'Education']])
# sns.pairplot(fit_data[["Age", 'Income']])
# sns.pairplot(fit_data[["Age", 'Usage']])
# sns.pairplot(fit_data[["Age", 'Fitness']])
# sns.pairplot(fit_data[["Age", 'Miles']])
# sns.lmplot(x = 'Age', y= 'Miles', data=fit_data, fit_reg=False)
# sns.boxplot(data=fit_data)
# sns.set_style('whitegrid')
# sns.violinplot(x='Product', y = 'Age', data=fit_data) # , inner=None => Removes bars inside the violins # TM195 is mostly used by customers aged 25, TM498 by customers aged 25 and 32, and TM798 by customers aged 36. The violiin plot shows the distribution through the thickness at the point
# Great for visualizing distributions
# sns.swarmplot(x='Product', y = 'Age', data=fit_data) # , color='k' => Makes the dots black # shows each point in the distribution and stacks those with similar values
# violin ans swarm plots are similar but differ in displaying the distribution
# melted_df = pd.melt(fit_data,id_vars=['Product', 'Usage', 'Miles'], var_name='Stat')
# print(melted_df.head())
# print(melted_df.shape)
# heatmaps -> to visualize matrix-like data
# corr = fit_data.corr()
# sns.heatmap(corr,annot=True) # => Miles+Fitness, Miles+Usage, Fitness+Usage, Income+Education are highlt correlated w/ values 0.79, 0.76, 0,67 and 0.63 resply
# sns.countplot(x='Product', data=fit_data) # barplots -> to visualize categorical vars
# plt.xticks(rotation=45)
# sns.countplot(fit_data['Product'], hue=fit_data['Product']) # Shows that there are 80 TM195, 60 TM498, and 40 TM798
# sns.factorplot(x='Product', y='Usage', data=fit_data)
# sns.kdeplot(fit_data['Fitness'], fit_data['Miles']) # displays distribution of 2 variables
# sns.jointplot(fit_data['Fitness'], fit_data['Miles']) # combines info from scatter plots and histogram => bi-variate distribution
# sns.countplot(fit_data['Gender'], hue=fit_data['Gender']) # Shows that there are approx 105 males and 75 females
# sns.countplot(fit_data['MaritalStatus'], hue=fit_data['MaritalStatus']) # Shows that there are approx 75 singles and 105 with spouses/partners

# sns.boxplot(fit_data['Product'], fit_data['Age'], hue=fit_data['Product']) # TM195 is used by customers in the age range of 24 to 33 with a median age of 26; Customers age could range from 17 to 47 and has one outlier with a customer ages 50 using TM195
# TM498 is used by customers in the age range of 24 to 33 with a median age of 26; Customers age could range from 19 to 45 and has one outlier with a customer age of 48 using TM498
# TM798 is used by customers in the age range of 25 to 30 with a median age of 27; Customers age could range from 22 to 38 and has 5 outliers with a customer ages of 40, 42, 45, 47, and 48 using TM798
# sns.pairplot(fit_data[['Age','Usage', 'Fitness', 'Miles']], kind="scatter")
# print(fit_data['Age'].max())
# sns.relplot(x="Product", y="Age", data=fit_data, style="Gender", hue="Education")
# sns.catplot(x='Product', y='Age',hue='MaritalStatus', data=fit_data,kind='bar') # box, point, bar, count, violin, strip
# plt.show()

# Multivariate Data Analysis


# print((fit_data.head(5))) # Displays the first 5 rows in the dataset
# sns.countplot(fit_data['Age'], hue=fit_data['Gender'])
# sns.boxplot(fit_data['Gender'], fit_data['Age'], hue=fit_data['Gender']) # shows that there is an outlier for the female group with one female of age 50
# sns.swarmplot(fit_data['Gender'], fit_data['Age'], hue=fit_data['Gender']) # More no of males aged 25 use the product more than any other customer of different age
# sns.jointplot(x="Age",y="Fitness", data=fit_data, kind='scatter')
# sns.distplot(fit_data['Education'], kde=True, rug=True)
# sns.barplot(x="Product",y="Fitness", data=fit_data,hue="Gender") # Customers (male and female) with self rated fitness score of mid value use the products TM195 and TM498
# sns.barplot(x="Product",y="Fitness", data=fit_data) # W.r.t the self rated fitness score, most of the 'very fit' customers use the TM798 product
# sns.countplot(fit_data["Fitness"]) # Approx 99 customers have self rated fitness of 3
# sns.boxplot(x="Age", y="Miles",data=fit_data)
# plt.show()

# honey_data = pd.read_csv("D://PythonPrgs/csvFiles/CardioGoodFitness.csv")
# print(honey_data.head())
# print(honey_data.shape)
colNames = ['Userid','Username', 'FirstName',	'LastName',	'Pwd',	'Cpwd',	'Gender','Dob','Place_of_birth','Location','Country','Zip_Code','Email','Phone','Hospital','College',
            'Qualification','Specialization','Designation','Department','Experience','Registration','Code	','Mode_of_Contact','Homepage_link','Income','Height','Weight',
            'Hospital','Vehicle	','Vehicle_no','MaritalStatus']

df = pd.DataFrame(columns=colNames)
# def rand_assign(lst):
#   randVals = np.random.choice(lst,size=len(df))
#   return randVals
#
# df["rand_day"] = df.apply(lambda row: random_day())
unames = pd.Series(['Madhura', 'Mamata', 'Mamta', 'Mahesh', 'Nidhi', 'Shriya', 'Poorvi','Sangmesh', 'Shridhar', 'Rashmi', 'Sunanda','Sadashiv', 'Priya', 'Nandini'])
# day_list = pd.to_datetime(['2015-01-02','2016-05-05','2015-08-09'])
# #alternative
# #day_list = pd.DatetimeIndex(['2015-01-02','2016-05-05','2015-08-09'])


for i in range(0, 1000):
    df.loc[i,'Username']= np.random.choice(unames)
print(df['Username'])


