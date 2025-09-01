# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
# Getting the data
from fuzzywuzzy import fuzz
a = fuzz.partial_ratio("this is a test", "this a test")
print(a)
b = fuzz.ratio("this is a test", "this is a")
print(b)
dataset = pd.read_csv("D:/Pythonprgs/csvFiles/Automobile.tra")
print(dataset.head())
dataset.to_csv("D:/Pythonprgs/csvFiles/Automobile_df.csv")

#Data Preprocessing
# #Checking for missing values
# ds = dataset.isna().sum()
# print(dataset.dtypes)
# lst = dataset.columns
# for i in lst:
#     print(i, " ", dataset[i].unique(), " ",  len(dataset[i].unique()))
# X = dataset.drop(['Year','Class','DataSource', 'Topic', 'Data_Value_Unit', 'StratificationCategory2', 'TopicID'],axis=1)
# print(X.head())
# lb_LocationAbbr = LabelEncoder()
# X["LocationAbbr"] = lb_LocationAbbr.fit_transform(X["LocationAbbr"])
# lb_LocationDesc = LabelEncoder()
# X["LocationDesc"] = lb_LocationDesc.fit_transform(X["LocationDesc"])
# lb_GeographicLevel = LabelEncoder()
# X["GeographicLevel"] = lb_GeographicLevel.fit_transform(X["GeographicLevel"])
# lb_Data_Value_Type = LabelEncoder()
# X["Data_Value_Type"] = lb_Data_Value_Type.fit_transform(X["Data_Value_Type"])
# # lb_Data_Value_Footnote_Symbol = LabelEncoder()
# # X["Data_Value_Footnote_Symbol"] = lb_Data_Value_Footnote_Symbol.fit_transform(X["Data_Value_Footnote_Symbol"])
# # lb_Data_Value_Footnote = LabelEncoder()
# # X["Data_Value_Footnote"] = lb_Data_Value_Footnote.fit_transform(X["Data_Value_Footnote"])
# lb_StratificationCategory1 = LabelEncoder()
# X["StratificationCategory1"] = lb_StratificationCategory1.fit_transform(X["StratificationCategory1"])
# lb_Stratification1 = LabelEncoder()
# X["Stratification1"] = lb_Stratification1.fit_transform(X["Stratification1"])
# lb_Stratification2 = LabelEncoder()
# X["Stratification2"] = lb_Stratification2.fit_transform(X["Stratification2"])
# # lb_Location1= LabelEncoder()
# # X["Location1"] = lb_Location1.fit_transform(X["Location 1"])
# print("NA",dataset.isna().sum())
# # print(X.head())
# print("Dropping na", X.dropna())
# print("Shape of dataset", X.shape)
#
# # ', '', '', '', '', '', '', 'Stratification2', '']
#
# # Encoding the categorical values
# # colNames = ['id','duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class', 'no', 'AttCat']
# # DosAtt = ["back", "land" , "neptune", "pod", "smurf", "teardrop", "udpstorm", "apache2", "processtable", "mailbomb"]
# # ProbeAtt = ["ipsweep", "nmap", "portsweep", "satan","saint", "mscan"]
# # u2rAtt = ["buffer_overflow", "loadmodule", "perl", "rootkit", "httptunnel","xterm", "ps", "sqlattack"]
# # r2lAtt = ["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient", "warezmaster", "snmpgetattack", "snmpguess","named", "sendmail", "worm", "xlock", "xsnoop"]
# # dataset['AttCat'] = ["Dos" if x in DosAtt else "Probe" if x in ProbeAtt else "U2R" if x in u2rAtt else "R2L" if x in r2lAtt else "normal" for x in dataset['class']]
# # print(dataset.head())
# # print(dataset['AttCat'].unique())
# # attacks = ["normal", "Dos", "Probe","U2R", "R2L"]
# #
# # #Splitting the dataset into train and test datasets
# # from sklearn.model_selection import train_test_split
# # X = dataset.drop('class', axis=1)
# # Y = dataset['class']
# # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
# #
# # # Data Normalization using StandardScaler
# #
# # from sklearn.ensemble import  RandomForestRegressor
# # regre = RandomForestRegressor(n_estimators=10, random_state= 0)
# # # regre.fit(X_train.reshape(-1,1), Y_train.reshape(-1, 1  ))