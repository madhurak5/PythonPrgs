import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
ds = pd.read_csv("C://PythonPrgs/csvFiles/cic/Friday_WorkingHours_Afternoon_PortScan_pcap_ISCX.csv")
print(ds.head())
print(ds.dtypes)
print(ds.shape)
lab = LabelEncoder()
ds.replace([np.inf, -np.inf], np.nan, inplace=True)
X_enc = pd.DataFrame(ds)
X_enc['Label'] = lab.fit_transform(X_enc['Label'])
y = X_enc.loc[:,'Label']
X_enc = X_enc.drop('Label', axis=1)
print(y)
X = X_enc
X_scaler = MinMaxScaler().fit(X_enc)
X = X_scaler
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# from sklearn.naive_bayes import BernoulliNB
# nb = BernoulliNB()
# # nb.fit(X, y)
# nbModel = nb.fit(X_train, y_train)
# gnbPred = nb.predict(X_test)
# print("----------------- Naive Bayes -----------------")
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# acc = accuracy_score(y_test, gnbPred)
# misclass = 1 - acc
# print("Accuracy Score : ", acc)
# print("Precision Score : ", precision_score(y_test, gnbPred, average='macro'))
# print("Recall Score : ", recall_score(y_test, gnbPred, average='macro'))
# print("F-score : ", f1_score(y_test, gnbPred, average='macro'))
# cmg = confusion_matrix(y_test, gnbPred)
# print("Confusion Matrix : \n", cmg)
