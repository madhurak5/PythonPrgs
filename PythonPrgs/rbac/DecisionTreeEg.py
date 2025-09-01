import pandas as pd
import math
import numpy as np
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
def assignRole():
    perData = "D:/PythonPrgs/csvFiles/credForm2.csv"
    df = pd.read_csv(perData)
    print(df)
    cols = df.columns
    pd.set_option('display.max_columns', None)
    print(cols)
    neceCols = ['username', 'Fname', 'Lname', 'Pwd', 'CPwd', 'Gender', 'DoB','Qualification', 'Designation', 'Experience', 'Department', 'Email','Role']
    print(df[neceCols].dtypes)
    lbl = LabelEncoder()
    for i in cols:
        if df[i].dtypes == object:
            df[i] = lbl.fit_transform(df[i])
            print(i)
        if df[i].isnull().values.any() :
            df[i] = 0
            # df['your column name'].isnull().values.any()
    print(df)
    y = df['Role']
    X = df.drop('Role', axis = 1 )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=41)
    print(X_train)
    print(X_test)
    model = tree.DecisionTreeClassifier()
    model.fit(X_train,y_train)
    print(model)
    y_pred = model.predict(X_test) # Prediction
    print(model.score(X_test, y_test))
    roleToBAssigned = model.predict([[0,0,0,0,0, 0,0,2,0,17, 0,0 ,1, 0]])
    roleToBeA = lbl.inverse_transform(roleToBAssigned)
    # print(roleToBeA[0])
    return (roleToBeA[0])

c = assignRole()
print(" Role assigned : ", c)
# from charm.toolbox.pairinggroup import PairingGroup
# from charm.core.math.pairing import pairing

# pg = PairingGroup.random(_type=1,count=1, seed=None)
# print(pg)
# print(c[0])
# # print((type(df)))
# # print(type(df.values))
# # DosAtt = ["back", "land" , "neptune", "pod", "smurf", "teardrop", "udpstorm", "apache2", "processtable", "mailbomb"]
# # ProbeAtt = ["ipsweep", "nmap", "portsweep", "satan","saint", "mscan"]
# # u2rAtt = ["buffer_overflow", "loadmodule", "perl", "rootkit", "httptunnel","xterm", "ps", "sqlattack"]
# # r2lAtt = ["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient", "warezmaster", "snmpgetattack", "snmpguess","named", "sendmail", "worm", "xlock", "xsnoop"]
# # df['AttCat'] = ["Dos" if x in DosAtt else "Probe" if x in ProbeAtt else "U2R" if x in u2rAtt else "R2L" if x in r2lAtt else "normal" for x in df['class']]
# # # attacks = ["back", "land" , "neptune", "pod", "smurf", "teardrop", "ipsweep", "nmap", "portsweep",
# # #            "satan", "buffer_overflow", "loadmodule", "perl", "rootkit","ftp_write", "guess_passwd",
# # #            "imap", "multihop", "phf", "spy", "warezclient", "warezmaster" ]
# # attacks = ["normal", "Dos", "Probe","U2R", "R2L"]
# # lb_proto = LabelEncoder()
# # lb_serv = LabelEncoder()
# # lb_flag= LabelEncoder()
# # lb_AttCat = LabelEncoder()

# # objFeatures = ['protocol_type', 'service','flag','AttCat']
# # df["protocol_type"] = lb_proto.fit_transform(df["protocol_type"])
# # df["service"] = lb_serv.fit_transform(df["service"])
# # df["flag"] = lb_flag.fit_transform(df["flag"])
# # df["AttCat"] = lb_AttCat.fit_transform(df["AttCat"])
# # df["class"] = lb_class.fit_transform(df["class"])
# # for i in objFeatures:
# #     print(df[i].unique())
# y = df['Role']
# X = df.drop('Role', axis = 1 )
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=41)
# # # print("\nX_train : \n")
# print(X_train)
# print(X_test)
# print("Shape of X_train : ",X_train.shape)
# print("\nX_test : \n")
# print(X_test.head())
# print("Shape of X_test : ", X_test.shape)
# model = tree.DecisionTreeClassifier()
# model.fit(X_train,y_train)
# print(model)
# print(df.dtypes)
# # print(df.drop(["PassengerId"],axis=1))
# # colNames = ["PassengerId","Survived","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]
# colNamesDrop = ["PassengerId","Name","SibSp","Parch","Ticket","Cabin","Embarked"]
# # newTitanic = df.drop(["PassengerId","Name","SibSp","Parch","Ticket","Cabin","Embarked"], axis=1)
# newTitanic = df.drop(colNamesDrop, axis=1)
# # newTitanic = df.drop(["PassengerId"])
# print(newTitanic.head())
# lbSex = LabelEncoder()
# newTitanic["Sex"] = lbSex.fit_transform(newTitanic["Sex"])
# print(newTitanic.dtypes)
# print(newTitanic.head(10))
# med_age = np.ceil(np.nanmedian(newTitanic["Age"]))
# mean_age = np.floor((np.mean(newTitanic["Age"]))) # Can also use mean of the Age
# print("Mean age : ", mean_age)
# newTitanic["Age"].fillna(mean_age, inplace=True)
#
# j = 0
# for i in newTitanic["Survived"].values:
#     if i == 1:
#         j+=1
# print("No. of people survived (Original dataset): ", j)
# notSurvived = newTitanic.shape[0] - j
# print("No. of people who dint survive (Original dataset : ", notSurvived)
# y = newTitanic["Survived"]
# print("Length of y : ", len(y))
# X = newTitanic.drop('Survived', axis = 1 )
#
# y = df["Survived"]
# scatter_matrix(df, c = y, figsize=[5,5], s=891, marker='D')
# plt.show()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=41)
# model = tree.DecisionTreeClassifier()
# # model = RandomForestClassifier()
# model.fit(X_train,y_train)  # Model fitting
# y_pred = model.predict(X_test) # Prediction
# # #
# print(model.score(X_test, y_test))
# cls = model.predict([[0,0,0,0,1, 0,0,0,0,0, 0,1 ,0,1]])
# print(lbl.inverse_transform(cls))
# le.inverse_transform([2, 2, 1])
# # siya,Siya,,siy,siy, Female,,M.Tech,Assoc. Professor,1,Electronics,siya@gmail.com,Admin,1,0
# # # print(y_test, " - ", y_pred)
# # cntSur = np.count_nonzero(y_train)
# # # print(y_train)
# # print("No. survived y_train: ",cntSur)
# # cntSury = np.count_nonzero(y_test)
# # print("cntSur y_test : ", cntSury)
# # # cntNotSur = np.count_
# # j = 0
# # for i in y_train.values:
# #     if i == 0:
# #         j+=1
# # print("Nonsurviving: ", j)
# #
# # j = 0
# # for i in y_test.values:
# #     if i == 0:
# #         j+=1
# # print("Nonsurviving(y_test): ", j)
# #
# print(model.predict(X_test))
# print(cross_val_score(model, X_test,y_test, cv = 10))
# from sklearn.metrics import mean_squared_error, r2_score
# mse = mean_squared_error(y_test, y_pred)
# print("Mean Square Error : ", mse)
# print("Root Mean Square Error : ", np.sqrt(mse))
# print("R2 score : ", r2_score(y_test, y_pred))
#
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# post = pca.fit_transform(newTitanic)
# post_df = pd.DataFrame(data = post, columns=['PC1', 'PC2'])
# print("Last 5 records from post_df : \n", post_df.tail())
# print("Shape after PCA ", post.shape)
# print("PCA Components : \n", pca.components_)
# print("Explained Variance ratio : ", pca.explained_variance_ratio_)
# import matplotlib.pyplot as plt
# plt.figure()
# plt.figure(figsize=(10, 10))
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=14)
# plt.xlabel("PC - 1 ", fontsize=20)
# plt.ylabel("PC - 2 ", fontsize=20)
# plt.title("Prnicipal Component Analysis of Titanic Dataset", fontsize=20)
# df['Survived'].replace(0, 'NS',inplace=True)
# df['Survived'].replace(1, 'S',inplace=True)
# targets = ['NS', 'S']
# colors = ['r', 'g']
# for targets, color in zip(targets,colors):
#     indicesToKeep = df['Survived'] == targets
#     plt.scatter(post_df.loc[indicesToKeep, 'PC1']
#                , post_df.loc[indicesToKeep,  'PC2'], c = color, s = 50)
#
# plt.show()