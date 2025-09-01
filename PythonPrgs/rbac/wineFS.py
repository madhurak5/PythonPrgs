import pandas as pd
# import
df = pd.read_csv("D:/PythonPrgs/csvFiles/wine.csv")
print(df.head ())
import matplotlib.pyplot as plt
print(df['Wine'].unique())
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df.drop(labels=['Wine'], axis=1),df['Wine'],test_size = 0.3, random_state=0)
print(X_train.head ())
from sklearn.feature_selection import  mutual_info_classif, RFE
mutual_info = mutual_info_classif(X_train, Y_train)
print(mutual_info)
mutual_info = pd.Series(mutual_info)
# print(mutual_info)
mutual_info.index = X_train.columns
# print(mutual_info.sort_values(ascending=False))
mutual_info.sort_values(ascending=False).plot.bar(figsize = (20, 8))

from sklearn.feature_selection import SelectKBest
sel_five_cols = SelectKBest(mutual_info_classif, k=5)
sel_five_cols.fit(X_train,Y_train)
print(X_train.columns[sel_five_cols.get_support()])

from sklearn.feature_selection import VarianceThreshold
var_thresh = VarianceThreshold(threshold=0.1)

var_thresh.fit(X_train)
const_cols = [colu for colu in X_train.columns if colu not in X_train.columns[var_thresh.get_support()]]
print(X_train.columns[var_thresh.get_support()])
print("Const cols")
for c1 in const_cols:
    print(c1)
print(const_cols)
from sklearn.feature_selection import chi2
fpvals = chi2(X_train, Y_train)
print("Fp Vals", fpvals)
pvals = pd.Series (fpvals[1])
pvals.index=X_train.columns
print(pvals)
print(pvals.sort_values(ascending=True))

from sklearn.tree import DecisionTreeClassifier
rfe = RFE(estimator=DecisionTreeClassifier(),n_features_to_select=8)
print("Rfe : \n",rfe)
modelrfe = DecisionTreeClassifier()
from sklearn.pipeline import Pipeline
pipeline = Pipeline(steps=[('s',rfe),('m',modelrfe)])
pipeline.fit(X_train,Y_train)
# data = [[14.23, 13.2 , 13.16 ,14.37 ,13.24, 14.2 , 14.39, 14.06 ,14.83, 13.86 ,14.1  ,14.12, 1.2]]
# data = [[12.86, 1.35 , 2.32,18.0,122, 1.51, 1.25, 0.21,0.94, 4.10 ,0.76,1.29, 1065]]
data = [[14.23, 1.71 , 2.43 ,15.6 ,127 ,  2.80  , 3.06, 0.28 , 2.29 , 5.64  ,1.04  ,3.92 , 1065]]
yhat = pipeline.predict(data)
print("Predicted class : ", yhat)
print("ROw 1 in dataset")
print(X_train.shape[1])
cols1 = df.columns
print(cols1[1])
for i in range(X_train.shape[1]):
    print("Column: %s, Selected %s, Rank: %.3f"%(cols1[i],rfe.support_[i], rfe.ranking_[i]))

# for i in df.columns:
#     print(df[i].values)
df_new = df[df['Wine'] == 1]
# pd.set_option('display.max_columns', None)
# print(df_new)

plt.figure(figsize=(12, 10))
cor = X_train.corr()

def correln(dataset, threshold):
    col_corr = set()
    corr_mat = dataset.corr()
    for i in range(len(corr_mat.columns)):
        for j in range(i):
            if abs(corr_mat.iloc[i,j]) > threshold:
                colname = corr_mat.columns[i]
                col_corr.add(colname)
    return col_corr


corr_features = correln(X_train,0.7)
print(len(set(corr_features)))
import seaborn as sns
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
plt.show()

print("sssssssssssssssssssss")
# from sklearn.datasets import load_boston
# data = load_boston()
# dfb = pd.DataFrame(data.data, columns=data.feature_names)
# dfb["MEDV"] = data.target
# print(data.feature_names)
# print(dfb.head())
# X = dfb.drop("MEDV", axis=1)
# Y = dfb["MEDV"]
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# plt.figure(figsize=(12, 10))
# cor = X_train.corr()
#
# def correln(dataset, threshold):
#     col_corr = set()
#     corr_mat = dataset.corr()
#     for i in range(len(corr_mat.columns)):
#         for j in range(i):
#             if abs(corr_mat.iloc[i,j]) > threshold:
#                 colname = corr_mat.columns[i]
#                 col_corr.add(colname)
#     return col_corr
#
#
# corr_features = correln(X_train,0.7)
# print(len(set(corr_features)))
# import seaborn as sns
# sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
# plt.show()
# print(corr_features)
# X_train.drop(corr_features, axis=1)
# X_test.drop(corr_features, axis=1)
# print(X_train.columns)
# print(X_test.columns)
#
# df = pd.read_csv("D:/PythonPrgs/csvFiles/mobileData.csv")
# print(df.head ())

