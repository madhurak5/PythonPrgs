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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score,classification_report, confusion_matrix,recall_score, roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.feature_selection import RFE, SelectKBest, VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')
data1 = pd.read_csv("D://PythonPrgs/csvFiles/userprofile.csv")
# print(data1.head())
print(data1.shape)
data2 = pd.read_csv("D://PythonPrgs/csvFiles/userpayment.csv")
# print(data2.head())
print(data2.shape)
data3 = pd.read_csv("D://PythonPrgs/csvFiles/usercuisine.csv")
# print(data3.head())
print(data3.shape)
# merge_data = pd.merge(data, data2, data3) # print("Merged dataset")  # print(merge_data.head()) # print(merge_data.shape)

# userrating_final.csv # usergeoplaces2.csv # userchefmozparking.csv # userchefmozhours4.csv # userchefmozcuisine.csv # userchefmozaccepts.csv
data_frames = [data1, data2, data3]
df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['userID'], how='inner'), data_frames)
df_merged1 = data1.merge(data2,on='userID',how='inner')
# print("Dfmerged1: \n", df_merged1.head())
df_merged2 = df_merged1.merge(data3, on='userID', how='inner')
# print("Dfmerged2: \n", df_merged2.shape)
userId_cnt = df_merged1.groupby(['userID'])['Upayment'].count().reset_index()
# print("userId_cnt: \n", userId_cnt['userID'].head())
#df_merged.to_csv("mergedFile1.csv")

newData = pd.read_csv("D://PythonPrgs/csvFiles/mergedFile1.csv")
# print(newData.head())
catCols = df_merged.select_dtypes(include=['object']).columns.tolist()
print("SMoker values: ", set(newData['personality']))
print(newData['personality'].value_counts())
print(newData.columns)
lab_enc = LabelEncoder()
for i in catCols:
    newData[i] = lab_enc.fit_transform(newData[i])

X = newData.drop(['uid','userID','Upayment'], axis=1)
print("X : -> \n", X.head())
y = newData['Upayment']
k = 10

from sklearn.preprocessing import label_binarize
# y = label_binarize(y, classes=[0,1])
# n_classes = y.shape[1]
mScaler = MinMaxScaler().fit(X)
X_scaler = mScaler.transform(X)
X = X_scaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()
nb = MultinomialNB() # BernoulliNB()
sv = svm.SVC()
import  this
print("This values : ",this.i)
models = [dt, lr, rf, nb, sv]
for i in models:
    i.fit(X_train, y_train)

    pred = i.predict(X_test)
    print(i)
    # print("Predictions : \n",pred)
    print("Accuracy Score : \n", accuracy_score(y_test, pred))
    # print("Precision Score : \n", precision_score(y_test, pred,average='weighted'))
    # print("F1 Score : \n", f1_score(y_test, pred,average='weighted'))
    # print("Classification Report : \n", classification_report(y_test, pred))
    # print("Confusion Matrix : \n", confusion_matrix(y_test, pred))

k = 8

df = pd.DataFrame(X, columns= ['%s'%i for i in range(X.shape[1])])
print("df cols : ", df.columns)
sel_feature = SelectKBest(chi2,k=k)
X_newd = sel_feature.fit(X, y)
print("X_newd shape", X_newd.shape)
print(df.columns[sel_feature.get_support()])
print("X_newd.scores_ : ",X_newd.scores_)
X_new_feat = pd.DataFrame(X_train)
# print("X_new_feat:\n", list(X_new_feat))
sel_feature_df = pd.DataFrame({'Features':list(X_new_feat), 'Scores':sel_feature.scores_})
# feature_selection = sel_feature_df[:,]
# print("feature selection :", feature_selection)
# print("sel feature df :\n", sel_feature_df.head())
sorted_sel_feat = sel_feature_df.sort_values(by='Scores', ascending=False)
# print("Features Sorted scorebased (only k):\n",list(sorted_sel_feat['Features']))
selkfeat = list(sorted_sel_feat['Features'])
# print("Only k features")
# for i in range(0, k):
#     print(selkfeat[i])
# print(len(sorted_sel_feat['Features']))
newX = sorted_sel_feat['Features']
# print("newX :\n", newX)
# print(sorted_sel_feat.head(k))

# for i in range(0, k):
#     chi_cols[i] = sorted_sel_feat[['Features']
# print("chi cols : \n", chi_cols)
# print("***********\n",sel_feature_df.head())
print("-------------------------------------------------------------------------------------------------------")
X_train_chi = sel_feature.transform(X_train)
print(X_train_chi.shape)
X_test_chi = sel_feature.transform(X_test)
print("X_train_chi shape: ", X_train_chi.shape)
for i in models:
    i.fit(X_train_chi, y_train)
    pred = i.predict(X_test_chi)
    print(i)
    # print("Predictions : \n",pred)
    score1 = cross_val_score(i, X_train_chi, y_train,cv=10)
    print("Score : ", score1.mean())
    print("Accuracy Score : \n", accuracy_score(y_test, pred))
    print("Precision Score : \n", precision_score(y_test, pred,average='weighted'))
    print("F1 Score : \n", f1_score(y_test, pred,average='weighted'))
    print("Recall Score : \n", recall_score(y_test, pred, average='weighted'))
    # print("Classification Report : \n", classification_report(y_test, pred))
    # print("Confusion Matrix : \n", confusion_matrix(y_test, pred))


# sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
# # sel = SelectFromModel(DecisionTreeClassifier(criterion='gini'))
# # sel = SelectFromModel(estimator=LogisticRegression())
# sel.fit(X_train, y_train)
# sel.get_support()
# X_new = pd.DataFrame(X_train)
# selected_feat= X_new.columns[(sel.get_support())]
# print("No of features selected : ", len(selected_feat))
# print("Selected features using SelectFromModel are : ",selected_feat)
# noofFeat = 10
# X_train_new = sel.transform(X_train)
# print("X_train_new:",X_train_new.shape)
# X_test_new= sel.transform(X_test)
# # for i in models:
# sel.fit(X_train_new, y_train)
# pred = sel.predict(X_test_new)
# print("SelectFrom Model Accuracy Score : \n", accuracy_score(y_test, pred))
# bestfeatures = SelectKBest(score_func=chi2, k=noofFeat)
# fit = bestfeatures.fit(X,y)
# dfscores = pd.DataFrame(fit.scores_)
# X_new1 = pd.DataFrame(X_train)
# featureScores = pd.concat([X_new1.columns,dfscores],axis=1)  # concat two dataframes for better visualization
# featureScores.columns = ['Specs','Score']   # nam
# print("Selected features using SelectKBest are : \n",featureScores.sort_values(by='Ranking'))

# -------------------------- Feature Selection using RFE ----------------------------------
# def featSelectionRFE():
from sklearn.feature_selection import RFE
# rfe_dt = DecisionTreeClassifier()
rfe_lr = LogisticRegression()
rfe_dt = OneVsRestClassifier(DecisionTreeClassifier())
rfe_rf = RandomForestClassifier()
rfe_nb = MultinomialNB()
rfe_sv = svm.SVC(kernel='linear')
est = rfe_rf
no_of_features = 5
# print("X_train columns : => \n", X_train.head())
rfe = RFE(estimator=est, step=1,n_features_to_select=no_of_features)
rfe.fit(X_train, y_train)
X_newr = pd.DataFrame(X_train)
selectedFeatures = pd.DataFrame({'Feature' : list(X_newr.columns), 'Ranking' : rfe.ranking_})
rankedFeatures = selectedFeatures.sort_values(by='Ranking',ascending=True)
sorted_sel_rfe = pd.DataFrame(rankedFeatures)
# print("Ranking:\n", selectedFeatures.sort_values(by='Ranking'))
print("Ranking:\n", )
ranks = pd.Series(selectedFeatures['Ranking'])
print("Ranks : \n",ranks)
print("X_newr shape: ", X_newr.shape)
if est == rfe_sv:
    print("SVM support: \n",rfe.support_)
    print("SVM ranking: \n", rfe.ranking_)
# print("Features Sorted scorebased (only k):\n",list(sorted_sel_feat['Features']))
sel_rfe = list(sorted_sel_rfe['Feature'])
print("RFE selected features : \n", sel_rfe)
feat_names = np.array(newData.columns)
print(list(feat_names))
print("Feature names :")
for i in range(0, len(sel_rfe)):
    print(feat_names[sel_rfe[i]], " ", rfe.ranking_[i])
# for i in range (0, len(sel_rfe)):
#     print(sel_rfe[i])
# print(sel_rfe[0:10])
# print(sorted_sel_rfe.head())
# X_train_rfe = pd.DataFrame(sorted_sel_rfe.loc[:7,['Feature']])
# print("X_train_rfe: \n",X_train_rfe)
# print(X_train_rfe.shape)
X_train_rfe = rfe.transform(X_train)
print("X_train_rfe shape: ", X_train_rfe.shape)
X_test_rfe = rfe.transform(X_test)
if est == rfe_dt:
    y_score = dt.fit(X_train_rfe, y_train)
    pred = dt.predict(X_test_rfe)

    print("RFE-DT Accuracy Score : \n", accuracy_score(y_test, pred))
    print("Precision Score : \n", precision_score(y_test, pred, average='weighted'))
    print("F1 Score : \n", f1_score(y_test, pred,average='weighted'))
elif est == rfe_lr:
    lr.fit(X_train_rfe, y_train)
    pred = lr.predict(X_test_rfe)
    print("RFE-LR Accuracy Score : \n", accuracy_score(y_test, pred))
    print("Precision Score : \n", precision_score(y_test, pred, average='weighted'))
    print("F1 Score : \n", f1_score(y_test, pred, average='weighted'))
elif est == rfe_rf:
    rf.fit(X_train_rfe, y_train)
    pred = rf.predict(X_test_rfe)
    print("RFE-Rf Accuracy Score : \n", accuracy_score(y_test, pred))
    print("Precision Score : \n", precision_score(y_test, pred, average='weighted'))
    print("F1 Score : \n", f1_score(y_test, pred, average='weighted'))
elif est == rfe_nb:
    rfe_nb.fit(X_train_rfe, y_train)
    pred = rfe_nb.predict(X_test_rfe)
    print("RFE-NB Accuracy Score : \n", accuracy_score(y_test, pred))
    print("Precision Score : \n", precision_score(y_test, pred, average='weighted'))
    print("F1 Score : \n", f1_score(y_test, pred, average='weighted'))
elif est == rfe_sv:
    rfe_sv.fit(X_train_rfe, y_train)
    pred = rfe_sv.predict(X_test_rfe)
    print("RFE-SVM Accuracy Score : \n", accuracy_score(y_test, pred))
    # print(rfe_sv.)
    print("Precision Score : \n", precision_score(y_test, pred, average='weighted'))
    print("F1 Score : \n", f1_score(y_test, pred, average='weighted'))


# def featSelectionKBest(noofFeat):
#     from sklearn.feature_selection import SelectKBest, chi2, f_classif
#     bestfeatures = SelectKBest(score_func=chi2, k=noofFeat)
#     fit = bestfeatures.fit(X,y)
#     dfscores = pd.DataFrame(fit.scores_)
#     # dfcolumns = pd.DataFrame(X.columns)
#     featureScores = pd.concat([X.columns,dfscores],axis=1)  # concat two dataframes for better visualization
#     featureScores.columns = ['Specs','Score']   # naming the dataframe columns
#     return bestfeatures, dfscores, featureScores
#
#
#
# noofFeat = 10
# bestfeatures, dfscores , featureScores = featSelectionKBest(noofFeat)
# feats = featureScores.loc[:15,'Specs']
# print(feats)
# selBestFeatures = featureScores.nlargest(noofFeat,'Score')
# # "{:.2f}".format(a_float)
# print("selBestFeatures .... \n", selBestFeatures)
# selBestColumns = selBestFeatures.iloc[:,0].values

# from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2
# # Information Gain
# imps = mutual_info_classif(X, y)
# feat_imp = pd.Series(imps, df_merged2.columns[0:len(df_merged2.columns)]).sort_values(ascending=False)
# print(feat_imp)
# feat_imp.plot(kind='barh', color = 'teal')
# # plt.show()
#
# # Chi square
# X_cat = X.astype(int)
# chi2_features = SelectKBest(chi2,k=4)
# kBest_features = chi2_features.fit_transform(X_cat, y)
# print("Orig feature no. : ", X_cat.shape[1]) # 21
# print("Reduced feature no. : ", kBest_features.shape[1]) # 4
#
# # Fisher's Score
# from skfeature.function.similarity_based import fisher_score
# ranks = fisher_score.fisher_score(X, y)
# feat_impsf = pd.Series(ranks, df_merged2.columns[0:len(df_merged2.columns)]).sort_values(ascending=False)
# print(feat_impsf)
# feat_impsf.plot(kind='barh', color='blue')
# # plt.show()

