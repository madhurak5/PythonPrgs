import random
from warnings import simplefilter

import cleverhans.tf2.attacks.fast_gradient_method
import cleverhans.tf2.attacks.carlini_wagner_l2
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

simplefilter(action='ignore', category=FutureWarning)
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import tensorflow

from keras.callbacks import EarlyStopping

# ------------------------ Writing to a file --------------------------------------
# def writeToFile(ipFile, OutFile):
#     line1 = ipFile.readline()
#     print(line1)
#     #
#     line1 = ipFile.readline()
#     print(line1)
#     outFile.write(line1)
#     id1 = ""
#     for i in dataset.values:
#         outFile.write(id1)
#         outFile.write("")
#         i.tofile(outFile, sep=",", format="%s")
#         outFile.write("\n")
        # id1 += 1
#     return
#
# ------------------------ Updating the column values --------------------------------------
#def updateCol(colValues, useCol):
# roles = {1:'admin', 2:'technician', 3:'superuser'}
# # # print(dataset.dtypes)
# for i in range(0, len(dataset)):
#     if(dataset.loc[i, 'class']=='normal'):
#         no = random.randint(1, 3)
#         dataset.loc[i, 'role'] = roles[no]
#     else:
#         dataset.loc[i, 'role'] = 'Invalid'

# ------------------------ Retrieving the data --------------------------------------
def getData(fName):
    fileName = "C://PythonPrgs/csvFiles/Files/"+fName+".csv"
    data = pd.read_csv(fileName)
    return data

# ------------------------ Retrieving attributes of the data --------------------------------------
def getAttributes(dataset):
    return (dataset.columns)

# ------------------------ Fitting and Making predictions with the model --------------------------------------
def modelFitPredict(mod, X_train, Y_train, X_test):
    model = mod.fit(X_train, Y_train)
    pred  = mod.predict(X_test)
    return model, pred

# ------------------------ Displaying Model Scores --------------------------------------
def modelScores(Y_test, pred):
    print("Confusion Matrix \n", confusion_matrix(Y_test, pred))
    print("Accuracy Score", accuracy_score(Y_test,pred))
    print("F-score ", f1_score(Y_test, pred))
    print("Precision : ", precision_score(Y_test, pred))
    print("Recall: ", recall_score(Y_test, pred))

# ------------------------ Changing Roles in a column --------------------------------------
# def changeRoles(dataset, roles, col):
#     for i in dataset[col].values:
#         row = random.randint(0, 22544)
#         if (dataset.loc[row, col] == 'Invalid'):
#             no = random.randint(1,4)
#             dataset.loc[row, col] = roles[no]
#     return dataset

# ------------------------ Writing to a CSV file --------------------------------------
def writeToCsv(dataset, file):
    dataset.to_csv("C://PythonPrgs/csvFiles/Files" + file + ".csv")
    print("Done")

# ------------------------ Printing dataset details --------------------------------------
def prnDsDetails(dataset):
    print(dataset.shape)
    print(dataset.head())

# ------------------------ Changing Role --------------------------------------
def changeRoles(dataset):
    noofRecs = len(dataset['duration'])-1
    print(random.randint(0, noofRecs))
    for i in range(0, noofRecs):
        row = random.randint(0, noofRecs)
        if(dataset.loc[row, 'role'] == 'Invalid'):
            no = random.randint(0,3)
            dataset.loc[row, 'role'] = roles[no]
    return (dataset)

# ------------------------ Creating Class Column --------------------------------------
def createClassCol(dataset):
    dataset['class'] = ''
    for i in range(0, len(dataset['role'])):
        if dataset.loc[i,'attType'] in attackTypes:
            dataset.loc[i,'class'] = "Anomaly"
        else:
            dataset.loc[i, 'class'] = "Normal"
    return (dataset)

# ----------------------------- Logistic Regression Model ---------------------------------
def logRegAlgo():
    from sklearn.linear_model import LogisticRegression
    logReg = LogisticRegression()
    logModel = logReg.fit(X_train, y_train)
    logPred = logReg.predict(X_test)
    print(accuracy_score(y_test, logPred))

# ----------------------------- Decision Tree Model ---------------------------------
def decisionTreeAlgo():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import  tree
    dtree = DecisionTreeClassifier()
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X,y)
    model, pred = modelFitPredict(dtree, X_train, y_train, X_test)
    # dtree.plot_tree(clf.fit(iris.data, iris.target))
    tree.plot_tree(clf.fit(X, y))
    # modelScores(Y_test, pred)
    print("Decision Tree : ",accuracy_score(y_test, pred))
    print(confusion_matrix(y_test, pred))
    # modelScores(Y_test, pred)
    return dtree
# ----------------------------- Random Forest Model ---------------------------------
def randomForestAlgo():
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=10, random_state=41)
    rfModel = rf.fit(X_train, y_train)
    rfPred = rf.predict(X_test)
    model,pred = modelFitPredict(rf, X_train, y_train, X_test)
    print("Random Forest : ",accuracy_score(y_test, pred))
    print(confusion_matrix(y_test, pred))
    # modelScores(Y_test, rfPred)

# ----------------------------- Naive Bayes Model ---------------------------------
def naiveBayesAlgo():
    from sklearn.naive_bayes import BernoulliNB
    gnb = BernoulliNB()
    gnbModel = gnb.fit(X_train, y_train)
    gnbPred = gnb.predict(X_test)
    print("Naive Bayes : ",accuracy_score(y_test, gnbPred))
    cm = confusion_matrix(y_test, gnbPred)
    print(cm)
    # sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False, xticklabels='auto', yticklabels='auto')
    # plt.xlabel("True Label")
    # plt.ylabel("Predicted Label")
    # plt.show()
    # modelScores(Y_test,gnbPred)
    # X_new =  [X_enc.loc[5,:]]
    # X_new = np.reshape([26,'tcp','private','S0',0,0,0,0,0,0,0,0,0,0,7,1.0,0.0,255,13,'neptune','admin','Anomaly'], (-1, 1))
    # X_new = [[26,'tcp','private','S0',0,0,0,0,0,0,0,0,0,0,7,1.0,0.0,255,13,'neptune','admin']]
    # y_new = gnb.predict(X_new)
    # print("y_new : ", y_new)

# -----------------------------K-means clustering on training data ---------------------------------
# def nearestNeigh(X_enc):
#
#
#     from sklearn.neighbors import NearestNeighbors
#     neigh = NearestNeighbors(n_neighbors=1)
#     nn = neigh.fit(X_enc)
#     print(nn)
#     a = neigh.radius_neighbors_graph(X_enc)
#     print(a)
# -----------------------------K-means clustering on training data ---------------------------------
# def kmeansAlgo():
#     from sklearn.cluster import KMeans
#     km = KMeans(n_clusters=4)
#     km.fit(X_train)
#     pred = km.predict(X_test)
#     print(accuracy_score(Y_test, pred))

# ----------------------------- Feature Selection using SelectFromModel ---------------------------------
# def featFromModel(model):
#     from sklearn.feature_selection import SelectFromModel
#     sfm = SelectFromModel(model, threshold=0.1, prefit=True)
#     X_sel = sfm.transform(X_enc)
#     impo = model.feature_importances_
#     indices = np.argsort(impo)[::]
#     # print(X_sel.shape)
#     # return X_sel

# ------------------------ To SelectKBest features from the dataset --------------------------------------
def featSelectionKBest(noofFeat):
    from sklearn.feature_selection import SelectKBest, chi2
    bestfeatures = SelectKBest(score_func=chi2, k=noofFeat)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    # plt.bar([i for i in range(len(fit.scores_))],fit.scores_)
    # plt.show()
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)  # concat two dataframes for better visualization
    featureScores.columns = ['Specs','Score']   # naming the dataframe columns
    # print(featureScores.nlargest(noofFeat,'Score'))  #print 10 best features
    return bestfeatures, dfscores, featureScores

# -------------------------- Feature Selection using RFE ----------------------------------
# def featSelectionRFE():
    # from sklearn.feature_selection import RFE
    # rfe = RFE(estimator=logReg, step=1)
    # rfe = rfe.fit(X_train, Y_train)
    # # # print(dataset.columns)
    # selectedFeatures = pd.DataFrame({'Feature ' : list(X_train.columns), 'Ranking' : rfe.ranking_})
    # print(selectedFeatures.sort_values(by='Ranking'))

# ----------------------------- Barplot ---------------------------------
# def showBarPlot():
    # sns.barplot(attCatCnt.index, attCatCnt.values, alpha = 0.9)
    # plt.title("Frequency")
    # plt.xlabel("Number of role")
    # plt.ylabel("role")
    # plt.show()

# ---------------------------- Predictions using single / series of inputs ----------------------------------
# def makePredictions():
    # for i in range(20, 30):
    #     X_new = [X_enc.loc[i, :]]
    #     y_new = gnb.predict(X_new)
    #     print("y_new : ",str(i) ,":",  y_new[0])

# --------------------------- PCA plot -----------------------------------
# def showPCAPlot():
    # from sklearn.decomposition import PCA
    # import matplotlib.pyplot as plt
    # from mlxtend.plotting import plot_pca_correlation_graph
    # # data = load_iris()
    # X, y = data.data, data.target
    # Y=X_enc.loc[:999, 'AttCat']
    # Y = y[:100]
    # plt.figure(figsize=(10,5))
    # X_pca = PCA().fit_transform(X_train)
    # plt.title('PCA - NSL-KDD dataset')
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    # plt.scatter(X_pca[:100,0],X_pca[:100,1],c=Y)
    # _ = plot_pca_correlation_graph(X_train,X_train.columns)
    # plt.show()

# -------------------------- Correlation map using Seaborn ------------------------------------
# def corrMatrixHeatmap():
    # from sklearn.preprocessing import OrdinalEncoder
    # AttCatLabels = OrdinalEncoder(categories=['normal','Dos','Probe','R2L','U2R'])
    # labs, unique = pd.factorize(AttCatLabels, sort=True)
    # print(labs)
    # plt.figure(figsize=(12,10))
    # cor = dataset.corr()

    # sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    # plt.show()

# -------------------------- Standard SCaler ------------------------------------
def stdscaling(dataset):
    scaler = StandardScaler().fit(dataset)
    X_scaler = scaler.transform(dataset)
    return X_scaler

# -------------------------- MinMax SCaler ------------------------------------
def minmaxscaling(dataset):
    minMaxScaler = MinMaxScaler().fit(dataset)
    X_scaler = minMaxScaler.transform(dataset)
    return X_scaler

newCols = ['protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
       'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
       'num_root', 'num_access_files', 'is_host_login', 'is_guest_login',
       'srv_count', 'serror_rate', 'rerror_rate', 'dst_host_count',
       'dst_host_srv_count','attType', 'role']

d1 = getData("rbacNSL_new")
DosRows = d1[d1['AttCat'] == 'Dos']
dosDf = pd.DataFrame(DosRows)
AttCatVals = {'normal':0,'Dos':1,'Probe':2,'U2R':3,'R2L':4}
d2 = d1
d2 = d2.drop(['id','no'],axis=1)
# d2.rename(columns={'class':'attType'}, inplace=True)
d2['class'] = ''
d2 = d2.drop('AttCat', axis=1)
roles = ['admin', 'superuser', 'technician','Invalid']
attackTypes = ["back", "land" , "neptune", "pod", "smurf", "teardrop", "udpstorm", "apache2", "processtable", "mailbomb","ipsweep", "nmap", "portsweep", "satan","saint", "mscan","buffer_overflow", "loadmodule", "perl", "rootkit", "httptunnel","xterm", "ps", "sqlattack","ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient", "warezmaster", "snmpgetattack", "snmpguess","named", "sendmail", "worm", "xlock", "xsnoop"]
finalClass = ('Normal', 'Anomaly')


dsNew1 = pd.read_csv("C://PythonPrgs/csvFiles/Files/newFile24AttCatRole.csv")

# dsNew1 = createClassCol(dsNew1)
# writeToCsv(dsNew1,"newFile24AttCatRoleClass")
#
# print(dsNew1['class'].value_counts())
dsNew = pd.read_csv("C://PythonPrgs/csvFiles/Files/newFile24AttCatRoleClass.csv")
dsNewOrig = dsNew.copy()

# catCols = dsNew.columns.
# pd.CategoricalDtype.is_dtype(dsNew.)
# X_enc = pd.get_dummies(dsNew, columns=['protocol_type','flag','service', 'attType', 'role'])
# X_enc = pd.get_dummies(dsNew, columns=['protocol_type'])
lab = LabelEncoder()
X_enc = pd.DataFrame(dsNew)
print(X_enc.head())
catCols = ['protocol_type','flag','service','AttCat','attType','role','class']
for i in catCols:
    X_enc[i] = lab.fit_transform(X_enc[i])
# print()
# X_enc = pd.get_dummies(dsNew, columns=['protocol_type','flag','service', 'attType', 'role'])
# X_enc = X_enc.drop('role', axis=1)
print("After encoding X--------------------------------------------------")
print(X_enc.head())
# y = X_enc.loc[:,'AttCat']
print("Hello World")
y = X_enc.loc[:,'class']
y1 = X_enc.loc[:,'class']
X_enc = X_enc.drop(['id1','id','class', 'no','AttCat'], axis=1)
X = X_enc

noofFeat = 24
print("Hello Reduced features ")
bestfeatures, dfscores , featureScores = featSelectionKBest(noofFeat)
feats = featureScores.loc[:15,'Specs']
# print("FEatures only : \n", feats)
# print("Best Features : \n", bestfeatures)
selBestFeatures = featureScores.nlargest(noofFeat,'Score')
# print("Selected Bst Features : \n", selBestFeatures.iloc[:,0])
selBestColumns = selBestFeatures.iloc[:,0].values
# print(selColumns)

k=0
selColumns = []
# myCols = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','num_failed_logins','logged_in','root_shell','su_attempted','num_root','num_file_creations','num_access_files','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','same_srv_rate','diff_srv_rate','role','class']
for i in selBestFeatures.iloc[:, 0]:
    selColumns.append(i)
# for i in range (0, len(selColumns)):
#     print(i, selColumns[i])
newDf = pd.DataFrame()
for i in getAttributes(X_enc):
    if i in selColumns:
        # print('P',i)
        newDf[[i]] = X_enc[[i]]
# print("New Df Columns before Scaling: \n",newDf.columns)
# X_enc = selColumns
# print(featureScores.nlargest(noofFeat,'Score'))  #print 10 best features

# X_scaler = stdscaling(X_enc)
# X_scaler = stdscaling(newDf)


X_scaler = minmaxscaling(X_enc)

X = X_scaler
# print("X columns: \n", newDf.columns)
# print(X)
# # featSelectionKBest(noofFeat)
X_train,X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=41)
# model =
naiveBayesAlgo()
print("*****************************LogRegModel***********************************")
logRegAlgo()
print("*****************************decisionTreeAlgo***********************************")
decisionTreeAlgo()
svmClf = svm.SVC(kernel='poly')
svmFit =svmClf.fit(X,y)
# print(svmFit)
svmPred = svmClf.predict(X_test)
# print(svmPred)
print("Accuracy Score of SVM 1: ", accuracy_score(y_test, svmPred))
from sklearn.metrics import f1_score, confusion_matrix

print("F-score of SVM 1: ", f1_score(y_test, svmPred, average='macro'))
print("Confusion Matrix of SVM 1: \n", confusion_matrix(y_test, svmPred))

# clf = decisionTreeAlgo()
# randomForestAlgo()
newDfm = pd.DataFrame(newDf)
newDfm['class'] = y1
# print("Original")
newSelCols = newDf.columns
# for i in range (0, len(newDf.columns)):
#     print(i, newSelCols[i])
newSelDf = pd.DataFrame()
for i in dsNewOrig.columns:
    if i in newSelCols:
        # print('P',i)
        newSelDf[[i]] = dsNewOrig[[i]]
# print(newSelDf.shape)
newDf['attType'] = dsNewOrig['attType']
newSelDf['AttCat'] = ''
newSelDf['attType'] = dsNewOrig['attType']
# print(newSelDf[['attType', 'AttCat']].head())
dos = ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop']
probe = ['ipsweep', 'nmap',  'portsweep', 'satan']
r2l = ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy', 'warezclient', 'warezmaster']
u2r = ['buffer_overflow', 'loadmodule', 'perl', 'rootkit']
# for i in range(0, len(newSelDf['class'])):
#     if newSelDf.loc[i, 'class'] == 'Anomaly' and newSelDf.loc[i, 'attType'] in dos:
#         newSelDf.loc[i,'AttCat'] = 'Dos'
#     elif newSelDf.loc[i, 'class'] == 'Anomaly' and newSelDf.loc[i, 'attType'] in probe:
#         newSelDf.loc[i, 'AttCat'] = 'Probe'
#     elif newSelDf.loc[i, 'class'] == 'Anomaly' and newSelDf.loc[i, 'attType'] in r2l:
#         newSelDf.loc[i, 'AttCat'] = 'R2L'
#     elif newSelDf.loc[i, 'class'] == 'Anomaly' and newSelDf.loc[i, 'attType'] in u2r:
#         newSelDf.loc[i, 'AttCat'] = 'U2R'
#     else:
#         newSelDf.loc[i, 'AttCat'] = 'Normal'
# print(newSelDf.head(15))
myNewDf = pd.read_csv("C://PythonPrgs/csvFiles/Files/finalFile.csv")

# work3 = pd.DataFrame(data=None, columns=myNewDf.columns)
# for i in range (0, len(myNewDf['role'])):
#     if myNewDf.loc[i, 'role'] in ['admin', 'superuser', 'technician'] and myNewDf.loc[i,'class'] == 'Anomaly':
#         work3 = work3.append(myNewDf.loc[i, :])
#
# work3 = work3.drop(['id','class'], axis=1)
# writeToCsv(work3,"admin3")

lab1 = LabelEncoder()
X_enc_new = pd.DataFrame(dsNew)
catCols_new = ['flag','service','attType','role','class']
for i in catCols_new:
    X_enc_new[i] = lab.fit_transform(X_enc_new[i])
# X_enc = X_enc.drop('role', axis=1)
# print(X_enc.head())
y = X_enc_new.loc[:,'AttCat']
# y = X_enc_new.loc[:,'class']
# y1 = X_enc_new.loc[:,'class']
X_enc_new = X_enc_new.drop(['id'], axis=1) #'attType','class','role'
X = X_enc_new
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=41)

# print(newSelDf['AttCat'].value_counts())
# writeToCsv(newSelDf, "finalFile")
mnb = BernoulliNB()

mnbModel = mnb.fit(X_train, y_train)
mnbPred = mnb.predict(X_test)
print("Naive Bayes Bernoulli : ",accuracy_score(y_test, mnbPred))
# print(confusion_matrix(y_test, mnbPred))
# from sklearn.tree import export_graphviz
# import pydotplus
# dot_data = StringIO()
# feat = list(X_enc.columns[0:])
# export_graphviz(dtree, out_file=dot_data, feature_names=feat, filled=True,rounded=True, special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('dtree2.png')


# nearestNeigh(X)

work3 = pd.read_csv("C://PythonPrgs/csvFiles/Files/admin3.csv")
# print(work3.head(20))
# print(work3.shape)
# print(work3['role'].value_counts())
# print(work3['AttCat'].value_counts())
# cnt = 0
# for i in range(0, len(work3['role'])):
#     if work3.loc[i, 'role'] == 'technician' and work3.loc[i,'AttCat'] in [ 'Dos'] :
#         cnt += 1
# print("No of Dos attacks by technician: ", cnt)
work3Cols = work3.columns
# print("Work3 Cols : \n", work3Cols)
works3NumCols = work3._get_numeric_data().columns
# print("Numeric Columns : \n", works3NumCols)
work3CatCols = list(set(work3Cols)-set(works3NumCols))
# print("Categorical Columns : \n", work3CatCols)
X_enc_w3 = pd.DataFrame(data=work3, columns=work3Cols)
lab = LabelEncoder()
for i in work3CatCols:
    # print(i)
    X_enc_w3[i] = lab.fit_transform(X_enc_w3[i])
# print(X_enc_w3.shape)
# print(work3['AttCat'].value_counts())
# # X_enc = X_enc.drop('role', axis=1)
# # print(X_enc.head())
y = X_enc_w3.loc[:,'AttCat']
# print(len(y))
# print("X len : ", len(X_enc_w3['AttCat']))
# print("y len : ", len(y['AttCat']))
X_enc_w3 = X_enc_w3.drop(['id', 'AttCat'], axis=1) #'attType','class','role'
X = X_enc_w3
print("_______________________ PART - I ___________________")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=41)
# print("X_train shape : ", X_train.shape)
# print("y_train shape : ", y_train.shape)
# print("X_test shape : ", X_test.shape)
# print("y_test shape : ", y_test.shape)
from cleverhans import tf2


model = Sequential()
model.add(Dense(10, input_dim=X.shape[1], activation='relu')) # kernel_initializer='normal',
# model.add(Dense(50, input_dim=X.shape[1], kernel_initializer='normal', activation='relu'))
# model.add(Dense(10, input_dim=X.shape[1], kernel_initializer='normal', activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# model.add(Dense(y.shape[1],activation='softmax'))
model.compile(loss='binary_crossentropy',optimizer='adam', metrics='accuracy')
hist = model.fit(X_train, y_train, epochs = 100, batch_size=1000, validation_split=0.2)
test_res = model.evaluate((X_test, y_test))
print(f'Test Resutls : Losss = {test_res[0]};    Accuracy ={test_res[1]*100}')
setattacks = cleverhans.tf2.attacks.fast_gradient_method.fast_gradient_method(model_fn=model, x = X_train, eps=0.1, norm=1)
print("Set of attacks ", setattacks)
cw_Attacks = cleverhans.tf2.attacks.carlini_wagner_l2.carlini_wagner_l2(model_fn=model, x = X_train)
# # mointor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=)
# model.fit(X_train, y_train, validation_split=(X_test, y_test), verbose=2, epochs=1000)
# pred = model.predict(X_test)
import numpy as np
from sklearn import metrics
# pred = np.argmax(pred, axis=1)
# y_eval = np.argmax(y_test, axis=1)
# score = metrics.accuracy_score(y_eval,pred)
# print("Validation Score of DL : {}".format(score))
naiveBayesAlgo()

svmClf = svm.SVC( kernel='sigmoid')
svm.SVC()
svmFit =svmClf.fit(X,y)
# print(svmFit)
svmPred = svmClf.predict(X_test)
# print(svmPred)
print("Accuracy Score of SVM sigmoid : ", accuracy_score(y_test, svmPred))
from sklearn.metrics import f1_score, confusion_matrix

print("F-score of SVM sigmoid : ", f1_score(y_test, svmPred,average='macro'))
print("Confusion Matrix of SVM sigmoid : \n", confusion_matrix(y_test, svmPred))

# decisionTreeAlgo()
# randomForestAlgo()


# mnb = decisionTreeAlgo()
# randomForestAlgo()
# mnbModel = mnb.fit(X_train, y_train)
# mnbPred = mnb.predict(X_test)
# print("Naive Bayes Bernoulli work3 : ",accuracy_score(y_test, mnbPred))
work4 = pd.read_csv("C://PythonPrgs/csvFiles/Files/1Dos3.csv")
print("Work 4 : \n", work4.shape, "\n", work4.head())
# print(work4['AttCat'].value_counts())
# work4 = work4.drop(['id1','id','AttCat'], axis=1)
work4Cols = work4.columns
# print("Work4 Cols : \n", work4Cols)
work4NumCols = work4._get_numeric_data().columns
# print("Numeric Columns : \n", work4NumCols)
work4CatCols = list(set(work4Cols)-set(work4NumCols))
# print("Categorical Columns : \n", work4CatCols)
X_enc_w4 = pd.DataFrame(data=work4, columns=work4Cols)
lab = LabelEncoder()
for i in work3CatCols:
    X_enc_w4[i] = lab.fit_transform(X_enc_w4[i])
print(X_enc_w4.shape)
# print(work4['AttCat'].value_counts())
# # X_enc = X_enc.drop('role', axis=1)
# # print(X_enc.head())
y = X_enc_w4.loc[:,'role']
print(len(y))
# print("X len : ", len(X_enc_w3['AttCat']))
# print("y len : ", len(y['AttCat']))
X_enc_w4 = X_enc_w4.drop(['id','AttCat','role'], axis=1) #'attType','class','role'
X = X_enc_w4
print("_______________________ PART - II ___________________")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=41)
# print("X_train shape : ", X_train.shape)
# print("y_train shape : ", y_train.shape)
print("Naive Bayes for only Dos attacks") # Accuracy : 0.3244
naiveBayesAlgo()
# decisionTreeAlgo()
# randomForestAlgo()
noofRecords = len(work4['AttCat'])
svmClf = svm.SVC(kernel='poly')
svmFit =svmClf.fit(X,y)
# print(svmFit)
svmPred = svmClf.predict(X_test)
# print(svmPred)
print("Accuracy Score of SVM : ", accuracy_score(y_test, svmPred))
from sklearn.metrics import f1_score, confusion_matrix
print("F-score of SVM : ", f1_score(y_test, svmPred, average='macro'))
print("Confusion Matrix of SVM : \n", confusion_matrix(y_test, svmPred))

# svmClf = svm.SVC(kernel='linear')
# svmFit =svmClf.fit(X,y)
# print(svmFit)
# svmPred = svmClf.predict(X_test)
# print(svmPred)

# print("Accuracy Score of SVM : ", accuracy_score(y_test, svmPred))
# from sklearn.metrics import f1_score, confusion_matrix
#
# print("F-score of SVM : ", f1_score(y_test, svmPred))
# print("Confusion Matrix of SVM : \n", confusion_matrix(y_test, svmPred))
# newDfDos = pd.DataFrame(data=None,columns=work4.columns)
# for i in range(0, len(work4['AttCat'])):
#     if work4.loc[i, 'AttCat'] == 'Dos' and work4.loc[i,'role'] in ['admin', 'superuser', 'technician']:
#         newDfDos = newDfDos.append(work4.loc[i, :])
# writeToCsv(newDfDos,"1Dos3")
# #
# newDfProbe = pd.DataFrame(data=None,columns=work4.columns)
# for i in range(0, len(work4['AttCat'])):
#     if work4.loc[i, 'AttCat'] == 'Probe' and work4.loc[i,'role'] in ['admin', 'superuser', 'technician']:
#         newDfProbe = newDfProbe.append(work4.loc[i, :])
# writeToCsv(newDfProbe,"2Probe3" )
# dfP = pd.read_csv("D://PythonPrgs/csvFiles/2Probe3.csv")
# print(dfP.shape, dfP['AttCat'].value_counts())
# newDfR2L = pd.DataFrame(data=None,columns=work4.columns)
# for i in range(0, len(work4['AttCat'])):
#     if work4.loc[i, 'AttCat'] == 'R2L' and work4.loc[i,'role'] in ['admin', 'superuser', 'technician']:
#         newDfR2L = newDfR2L.append(work4.loc[i, :])
# writeToCsv(newDfR2L,"3R2L3")
dfR = pd.read_csv("C://PythonPrgs/csvFiles/Files/3R2L3.csv")
print(dfR.shape, dfR['AttCat'].value_counts())

# newDfU2R = pd.DataFrame(data=None,columns=work4.columns)
# for i in range(0, len(work4['AttCat'])):
#     if work4.loc[i, 'AttCat'] == 'U2R' and work4.loc[i,'role'] in ['admin', 'superuser', 'technician']:
#         newDfU2R = newDfU2R.append(work4.loc[i, :])
# writeToCsv(newDfU2R,"4U2R3")
dfU = pd.read_csv("C://PythonPrgs/csvFiles/Files/4U2R3.csv")
print(dfU.shape, dfU['AttCat'].value_counts())

# print("NEw Df 4 : \n", newDfDos.shape,"\n",newDfDos.head())
# # for i in range(0, len(work4['AttCat'])):
# #     if work4.loc[i, 'AttCat'] == 'Dos' and work4.loc[i,'role'] in ['admin', 'superuser', 'technician']:
# #         newDfDos = newDfDos.append(work4.loc[i, :])
# #     elif work4.loc[i, 'AttCat'] == 'Probe' and work4.loc[i,'role'] in ['admin', 'superuser', 'technician']:
# #         newDfProbe = newDfProbe.append(work4.loc[i, :])
# #     elif work4.loc[i, 'AttCat'] == 'R2L' and work4.loc[i,'role'] in ['admin', 'superuser', 'technician']:
# #         newDfR2L = newDfR2L.append(work4.loc[i, :])
# #     elif work4.loc[i, 'AttCat'] == 'U2R' and work4.loc[i,'role'] in ['admin', 'superuser', 'technician']:
# #         newDfU2R = newDfU2R.append(work4.loc[i, :])
#
# print(newDfDos.head())
# print(newDfDos['AttCat'].value_counts())
# newDfDos = pd.read_csv("D://PythonPrgs/csvFiles/1Dos3.csv")
# # writeToCsv(newDfDos,"1Dos3")
# print(newDfDos.shape)
# print(newDfDos['role'].value_counts())
# newDfProbe = newDfProbe.drop('AttCat',axis=1)
# # writeToCsv(newDfProbe,"2Probe3")
# print(newDfProbe.shape)
# print(newDfProbe['role'].value_counts())
#
# newDfR2L = newDfR2L.drop('AttCat',axis=1)
# # writeToCsv(newDfR2L,"3R2L3")
# print(newDfR2L.shape)
# print(newDfR2L['role'].value_counts())
#
# newDfU2R = newDfU2R.drop('AttCat',axis=1)
# # writeToCsv(newDfU2R,"4U2R3")
# print(newDfU2R.shape)
# print(newDfU2R['role'].value_counts())
