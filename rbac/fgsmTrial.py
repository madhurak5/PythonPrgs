import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler

df = pd.read_csv("C://PythonPrgs/csvFiles/KddTrain_att17.csv")
print(df.head())
# dfAtt = df.loc[df['attCat'] != 'normal']
# print(dfAtt.head())
print(df.dtypes)
lab = LabelEncoder()
ohe = OneHotEncoder()

pro = df['protocol_type'].unique()
pro.sort()
dfEnc = pd.DataFrame(ohe.fit_transform(df[['protocol_type']]).toarray(), columns="protocol_type_"+pro)
finalDf = df.join(dfEnc)
flg = df['flag'].unique()
flg.sort()
dfEnc = pd.DataFrame(ohe.fit_transform(df[['flag']]).toarray(), columns="flag_"+flg)
finalDf = df.join(dfEnc)
ser = df['service'].unique()
ser.sort()
dfEnc = pd.DataFrame(ohe.fit_transform(df[['service']]).toarray(), columns="service_"+ser)
finalDf = df.join(dfEnc)
cls = df['class'].unique()
cls.sort()
dfEnc = pd.DataFrame(ohe.fit_transform(df[['class']]).toarray(), columns="class_"+cls)
finalDf = df.join(dfEnc)

finalDf = finalDf.drop(['protocol_type', 'flag','service', 'class'], axis=1)
print(finalDf.head())
finalDf['attCat'] = lab.fit_transform(finalDf['attCat'])
print(finalDf['attCat'])
# normal - 4; DoS - 0; Probe - 1; R2L - 2; U2R - 3;
# # icmp - 0; tcp - 1; udp - 2

attDf = finalDf.loc[finalDf['attCat']!=4]
attNor = finalDf.loc[finalDf['attCat']==4]
X = attDf.drop(['attCat'], axis = 1)
y = attDf['attCat']
scaler = MinMaxScaler().fit(X)
X_scaler = scaler.transform(X)
norScaler = MinMaxScaler().fit(attNor)
nor_Scaler = norScaler.transform(attNor)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.naive_bayes import GaussianNB, BernoulliNB
nb = GaussianNB()
nb.fit(X_train, y_train)
nbPred = nb.predict(X_test)
cm = confusion_matrix(y_test, nbPred)
print(cm)
print(attDf.shape)
accScore = accuracy_score(y_test, nbPred)
print("Accuracy Score : ", accScore)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
dt = LogisticRegression()# MLPClassifier(hidden_layer_sizes=50, activation='relu', solver='adam' )

dt.fit(X_train, y_train)
dtPred = dt.predict(X_test)
cm = confusion_matrix(y_test, dtPred)
print(cm)
accScore = accuracy_score(y_test, dtPred)
print("Accuracy Score : ", accScore)


