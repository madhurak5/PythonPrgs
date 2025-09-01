import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.naive_bayes import BernoulliNB, GaussianNB,MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score,classification_report, confusion_matrix,recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.feature_selection import RFE, SelectKBest, VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
from sklearn.svm import SVC
warnings.filterwarnings('ignore')
# pd.set_option('display.max_columns', 100)
# Department_Information, Employee_Information, Student_Counseling_Information, Student_Performance_Data
# demographic, diagnosis,encounter, measurement,  medication, procedure, procedure_surg_hx, sleep_study, sleep_enc_id
# dep_data = pd.read_csv("D://PythonPrgs/csvFiles/PhysioNet/procedure.csv")
# emp_data = pd.read_csv("D://PythonPrgs/csvFiles/PhysioNet/procedure_surg_hx.csv")
# stud_data = pd.read_csv("D://PythonPrgs/csvFiles/healthcare/test_data.csv")
# perform_data = pd.read_csv("D://PythonPrgs/csvFiles/healthcare/train_data.csv")
# perform_data = pd.read_csv("D://PythonPrgs/csvFiles/diabetic_data.csv")
patient_data = pd.read_csv("D://PythonPrgs/csvFiles/Train/Patient_Profile.csv")
def getDatasetDetails(ds):
    print(ds.shape)
    print(ds.head())
print(patient_data.isna().sum())

#
# print("Department_Information:")
# getDatasetDetails(dep_data)
# # for i in dep_data.columns:
# #     print(dep_data[i].value_counts())
# # print("Employee_Information:")
# getDatasetDetails(emp_data)
# # print("Student_Counseling_Information:")
# getDatasetDetails(stud_data)
# print("Student_Performance_Data:")
getDatasetDetails(patient_data)
print(patient_data.dtypes)
newData = pd.read_csv("D://PythonPrgs/csvFiles/Train/Patient_Profile.csv")
# print(newData.head())
newData['City_Type'].fillna(value=0, inplace=True)
newData['Employer_Category'].fillna(value=0, inplace=True)
# newData = newData.drop(['City_Type','Employer_Category'], axis = 1)
catCols = [x for x in newData.columns if newData[x].dtype == "object"]
print(catCols)
print(newData['City_Type'].value_counts())
print(newData['Employer_Category'].value_counts())
lab_enc = LabelEncoder()
for col in catCols:
    if col in newData.columns:
        i = newData.columns.get_loc(col)
        newData.iloc[:,i] = newData.apply(lambda i:lab_enc.fit_transform(i.astype(str)), axis=0, result_type='expand')
#
X = newData.drop(['Employer_Category'],axis = 1)
pd.set_option('display.max_columns', 100)
print("X : -> \n", X.head())
y = newData['City_Type']
mScaler = MinMaxScaler().fit(X)
X_scaler = mScaler.transform(X)
X = X_scaler
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
dt = LogisticRegression()
sv = MultinomialNB()#svm.SVC(kernel='linear')
sv.fit(X_train, y_train)
p2 = sv.predict([[0, 516956,0, 0,0,0,1,208,9,11]])
# p2 = dt.predict([[2, 508307,0, 0,0,0,3,185,16, 89]])
# print("p1 prediction : ", lab_enc.inverse_transform(p1))
# roleToBAssigned = model.predict([[0,0,0,0,0, 0,0,2,0,17, 0,0 ,1, 0]])
p2= lab_enc.inverse_transform(p2)
print(p2[0])
# print("p2 prediction : ", lab_enc.inverse_transform(p2))
pred = sv.predict(X_test)
print("Accuracy Score : \n", accuracy_score(y_test, pred))
print("Precision Score : \n", precision_score(y_test, pred,average='weighted'))
print("F1 Score : \n", f1_score(y_test, pred,average='weighted'))
print("Recall Score : \n", recall_score(y_test, pred, average='weighted'))

# # Merging Datasets
# # Department and Employee
# dep_emp_data = dep_data.merge(emp_data, on=['Department_ID'], how='inner')
# print("Merged Deparment Employee data :\n",dep_emp_data.head())
# print(dep_emp_data.shape)
# print(dep_emp_data.dtypes)
# stud_data['Department_ID'] = stud_data['Department_Choices']
# # stud_data= stud_data.drop(['Department_Choices','Department_Admission'],axis=1)
# print(stud_data.head())
# # dept_cnt = df_merged1.groupby(['Department_ID'])['Employee ID'].count().reset_index()
# # print(dept_cnt.head(34))
# # print(dept_cnt.shape)
#
#
# # stud_perform_data = stud_data.merge(perform_data, on="Student_ID", how='inner')
# stud_perform_data = stud_data.merge(perform_data, on='Student_ID') #, indicator=True)
# print("Merged Student Performance data :")
# getDatasetDetails(stud_perform_data)
# print(stud_perform_data.dtypes)
#
# final_data = dep_emp_data.merge(stud_perform_data, on="Department_ID")
# print("Final Merged Data ->")
# final_data.to_csv("D://PythonPrgs/csvFiles/College/finalCollegeData1.csv",index=False)
# getDatasetDetails(final_data)
# print(final_data.dtypes)
# X = final_data.drop(['Department_Name'], axis = 1)
# y = final_data['Department_Name']
# mScaler = MinMaxScaler().fit(X)
# X_scaler = mScaler.transform(X)
# X = X_scaler
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
# dt = DecisionTreeClassifier()
# dt.fit(X_train, y_train)
# pred = dt.predict(X_test)
# print("Accuracy Score : \n", accuracy_score(y_test, pred))
#
#
# catCols = final_data.select_dtypes(include=['object']).columns.tolist()
# lab_enc = LabelEncoder()
# for i in catCols:
#     final_data[i] = lab_enc.fit_transform(final_data[i])
# print(final_data.head())