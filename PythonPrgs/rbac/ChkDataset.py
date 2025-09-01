import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# cols = data.columns
# print("Columns in wdbc dataset are :\n",cols)
# catCols = data.select_dtypes(include=['object']).columns.tolist()
# noOfCatCols = len(catCols)
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# # ohe = OneHotEncoder(categories=[2])
# # data = ohe.fit_transform(data).toarray()
# # print(data.head())
# for i in catCols :
#     data[i] = LabelEncoder().fit_transform(data[i])
# #     data[i] = OneHotEncoder(categories=i).fit_transform(data[i]).toarray()
# # for col in data.columns:
# #     print(col, ':', len(data[col].unique()), " labels")
# # print(data['gender'].value_counts())
# # for i in cols:
# # pd.set_option('display.max_columns', None)
# # print(data.head())
#
# # print(data['gender'].value_counts()) #diabetesMed change readmitted
# # print(data.head())
# # print(data.shape)
# # print(data)

from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder, OneHotEncoder, MinMaxScaler
df = pd.read_csv("D://PythonPrgs/csvFiles/userprofile.csv")
# from sklearn.datasets import mall
# df =
print(df.columns)
print(df.describe())
print(df.shape)
print(df.isnull().sum())
pd.set_option('display.max_columns', None)
print(df.head())
import seaborn as sns

print(df['personality'].unique())
import matplotlib.pyplot as plt
# print(df.value_counts())
# df_rest = sns.load_dataset(df)
# print("df_rest : \n", df_rest.head())
# df['activity'].value_counts().plot(kind='barh');
# df['activity'].value_counts(normalize=True).plot(kind='bar')
#
print("Pivot structure")
print(df.groupby(['budget', 'activity'])['weight'].mean().unstack())
# df.groupby(['budget', 'activity'])['weight'].mean().plot(kind='bar')
# sns.set(font_scale=1.4)
# df_rest['sex'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Gender", labelpad=14)
plt.ylabel("Count of People", labelpad=14)
plt.title("Count of People Who Received Tips by Gender", y=1.02);
plt.show()
# df['activity'].value_counts().plot(kind='barh')
# df['activity'].value_counts().sort_values().plot(kind='barh')
# pd.DataFrame(df['activity'].value_counts()).plot()
# df.letters.value_counts().sort_values().plot(kind = 'barh')
for i in range(0, len(df['activity'])):
    if df.loc[i, 'activity'] in ['professional', 'working-class']:
        df.loc[i, 'class'] = "owner"
    else:
        df.loc[i, 'class'] = "consumer"
# df.loc[5, 'c1'] = 'Value'
# print(df.head())
# # print(df.tail())
# print(df.groupby(['budget', 'class'])['weight'].mean())
# df.groupby(['budget', 'class'])['weight'].mean().plot(kind='bar')
print(df['budget'].value_counts())
catCols = df.select_dtypes(include=['object']).columns.tolist()
print("Cat cols : \n", catCols)
for i in catCols :
    df[i] = LabelEncoder().fit_transform(df[i])
    # df[i] = OneHotEncoder(categories=i).fit_transform(df[i]).toarray()
print(df.head())
df_std = StandardScaler().fit_transform(df)
print(df_std)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df.drop(labels=['class'], axis=1),df['class'],test_size = 0.3, random_state=0)
print(X_train.head ())
from sklearn.feature_selection import  mutual_info_classif, RFE
mutual_info = mutual_info_classif(X_train, Y_train)
print("mutual_info: \n", mutual_info)
mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
print(mutual_info.sort_values(ascending=False))
mutual_info.sort_values(ascending=False).plot.bar(figsize = (20, 8))
#
from sklearn.feature_selection import SelectKBest
sel_five_cols = SelectKBest(mutual_info_classif, k=5)
sel_five_cols.fit(X_train,Y_train)
print(X_train.columns[sel_five_cols.get_support()])

from sklearn.feature_selection import VarianceThreshold
var_thresh = VarianceThreshold(threshold=0.1)
#
var_thresh.fit(X_train)
const_cols = [colu for colu in X_train.columns if colu not in X_train.columns[var_thresh.get_support()]]
print(X_train.columns[var_thresh.get_support()])
print("Const cols")
for c1 in const_cols:
    print(c1)
print(const_cols)
# from sklearn.feature_selection import chi2
# fpvals = chi2(X_train, Y_train)
# print("Fp Vals", fpvals)
# pvals = pd.Series (fpvals[1])
# pvals.index=X_train.columns
# print(pvals)
# print(pvals.sort_values(ascending=True))
from sklearn.linear_model import LinearRegression, Perceptron, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
rfe = RFE(estimator=RandomForestClassifier(),n_features_to_select=8)
print("Rfe : \n",rfe)
modelrfe = RandomForestClassifier() #
# # modelrfe = DecisionTreeClassifier()
# modelrfe = GradientBoostingClassifier()
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(modelrfe, X_train, Y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print("Scores of evaluation:", np.mean(scores), np.std(scores))
pipeline = Pipeline(steps=[('s',rfe),('m',modelrfe)])
pipeline.fit(X_train,Y_train)
# # data = [[14.23, 13.2 , 13.16 ,14.37 ,13.24, 14.2 , 14.39, 14.06 ,14.83, 13.86 ,14.1  ,14.12, 1.2]]
# # data = [[12.86, 1.35 , 2.32,18.0,122, 1.51, 1.25, 0.21,0.94, 4.10 ,0.76,1.29, 1065]]
# # data = [[14.23, 1.71 , 2.43 ,15.6 ,127 ,  2.80  , 3.06, 0.28 , 2.29 , 5.64  ,1.04  ,3.92 , 1065]]
# df = [[104, 18.886698, -99.114979 , 1,0,4,1,1,2,2,1930,1, 1,4, 2,3,60,3, 1.57]]
df = [[104, 22.120019, -100.950991, 1,2,2, 1,3,2,2,1991,24,1,1,2,2,40,3,1.74 ]]
yhat = pipeline.predict(df)
print("Predicted class : ", yhat)
import seaborn as sns
import matplotlib.pyplot as plt
tips = sns.heatmap(df)
# plt.show()
# print("ROw 1 in dataset")
# print(X_train.shape[1])
# cols1 = data.columns
# print(cols1[1])
# for i in range(X_train.shape[1]):
#     print("Column: %s, Selected %s, Rank: %.3f"%(cols1[i],rfe.support_[i], rfe.ranking_[i]))
#
# # for i in df.columns:
# #     print(df[i].values)
# # df_new = data[data['diabetesMed'] == 1]
# # pd.set_option('display.max_columns', None)
# # print(df_new)
#
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
# -----------------------------------------------------------------------------------------------------------------

# data = pd.read_csv("D://PythonPrgs/csvFiles/insurance.csv") # dataset_diabetes/diabetic_data.csv Department_Information D:\PythonPrgs\csvFiles\   #HortonGeneralHospital
# print(data.head())
# print("shape:", data.shape)
# print("Chkg missing values: \n",data.isna().apply(pd.value_counts))
# print("Description: \n",data.describe().T)
# print("Information about data:\n",data.info())
# plt.figure(figsize=(20, 15))
# plt.subplot(3,3,1)
# plt.hist(data.bmi,color='lightblue',edgecolor = 'black', alpha=0.7)
# plt.xlabel('bmi')
# plt.subplot(3,3,2)
# plt.hist(data.age, color='lightblue', edgecolor = 'black', alpha = 0.7)
# plt.xlabel('age')
# plt.subplot(3,3,3)
# plt.hist(data.charges, color='lightblue', edgecolor = 'black', alpha = 0.7)
# plt.xlabel('charges')
# plt.show()
# import statsmodels.api as sm
# import scipy.stats as stats
# import seaborn as sns
#
# import scipy.stats as stats
# Skewness = pd.DataFrame({'Skewness' : [stats.skew(data.bmi),stats.skew(data.age),stats.skew(data.charges)]},
#                         index=['bmi','age','charges'])  # Measure the skeweness of the required columns
# print(Skewness)
# plt.figure(figsize= (20,15))
# plt.subplot(3,1,1)
# sns.boxplot(x= data.bmi, color='lightblue')
#
# plt.subplot(3,1,2)
# sns.boxplot(x= data.age, color='lightblue')
#
# plt.subplot(3,1,3)
# sns.boxplot(x= data.charges, color='lightblue')
#
# plt.show()
#
# plt.figure(figsize=(20,25))
#
#
# x = data.smoker.value_counts().index    #Values for x-axis
# y = [data['smoker'].value_counts()[i] for i in x]   # Count of each class on y-axis
#
# plt.subplot(4,2,1)
# plt.bar(x,y, align='center',color = 'lightblue',edgecolor = 'black',alpha = 0.7)  #plot a bar chart
# plt.xlabel('Smoker?')
# plt.ylabel('Count ')
# plt.title('Smoker distribution')
#
# x1 = data.sex.value_counts().index    #Values for x-axis
# y1 = [data['sex'].value_counts()[j] for j in x1]   # Count of each class on y-axis
#
# plt.subplot(4,2,2)
# plt.bar(x1,y1, align='center',color = 'lightblue',edgecolor = 'black',alpha = 0.7)  #plot a bar chart
# plt.xlabel('Gender')
# plt.ylabel('Count')
# plt.title('Gender distribution')
#
# x2 = data.region.value_counts().index    #Values for x-axis
# y2 = [data['region'].value_counts()[k] for k in x2]   # Count of each class on y-axis
#
# plt.subplot(4,2,3)
# plt.bar(x2,y2, align='center',color = 'lightblue',edgecolor = 'black',alpha = 0.7)  #plot a bar chart
# plt.xlabel('Region')
# plt.ylabel('Count ')
# plt.title("Regions' distribution")
#
# x3 = data.children.value_counts().index    #Values for x-axis
# y3 = [data['children'].value_counts()[l] for l in x3]   # Count of each class on y-axis
#
# plt.subplot(4,2,4)
# plt.bar(x3,y3, align='center',color = 'lightblue',edgecolor = 'black',alpha = 0.7)  #plot a bar chart
# plt.xlabel('No. of children')
# plt.ylabel('Count ')
# plt.title("Children distribution")
#
# plt.show()
#
# import copy
# from sklearn.preprocessing import LabelEncoder
# df_encoded = copy.deepcopy(data)
# df_encoded.loc[:,['sex', 'smoker', 'region']] = df_encoded.loc[:,['sex', 'smoker', 'region']].apply(LabelEncoder().fit_transform)
#
# sns.pairplot(df_encoded)  #pairplot
# plt.show()


# -----------------------------------------------------------------------------------------------------------------

# import re
# text_to_search = '''
# abcdefghijklmnopqrstuvwxyz
# ABCDEFGHIJKLMNOPQRSTUVWXYZ
# uas.edu
# vtu.in
# abc
# Hello HelloHello
# 1232356436
# '''
#
# pat = re.compile(r'[123|hell]') # abc  cba
# matches = pat.finditer(text_to_search)
# for mat in matches:
#     print(mat.span(0))
#     print(mat.group(0))
    # print(text_to_search[mat.span(0)[0]:mat.span(0)[1]])

# -----------------------------------------------------------------------------------------------------------------
# sns.barplot()
# import numpy as np
# import nltk
# import string
# import random
# f = open('D://PythonPrgs/chatbot.txt', 'r', errors='ignore')
# raw_doc = f.read()
#
# raw_doc = raw_doc.lower()
# nltk.download('punkt')
# nltk.download('wordnet')
# sent_tokens = nltk.sent_tokenize(raw_doc)
# word_tokens = nltk.word_tokenize(raw_doc)
# print(sent_tokens[:2])
# print(word_tokens[:5])
# lemmer = nltk.stem.WordNetLemmatizer()
# def LemTokens(tokens):
#     return [lemmer.lemmatize(token) for token in tokens]
#
# remove_punct_dict = dict ((ord(punct), None) for punct in string.punctuation)
#
# def LemNormalize(text):
#     return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
#
# greet_input = ("hello", "hi", "greetings", "sup", "hey")
# greet_response = ["hi","hey","*nod*", "Glad that u r a talking to me"]
# def greet(sentence):
#     for word in sentence.split():
#         if word.lower() in greet_input:
#             return random.choice(greet_response)
#
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# def response(user_response):
#   robo1_response=''
#   TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
#   tfidf = TfidfVec.fit_transform(sent_tokens)
#   vals = cosine_similarity(tfidf[-1], tfidf)
#   idx=vals.argsort()[0][-2]
#   flat = vals.flatten()
#   flat.sort()
#   req_tfidf = flat[-2]
#   if(req_tfidf==0):
#     robo1_response=robo1_response+"I am sorry! I don't understand you"
#     return robo1_response
#   else:
#     robo1_response = robo1_response+sent_tokens[idx]
#     return robo1_response
#
# flag=True
# print("BOT: My name is Stark. Let's have a conversation! Also, if you want to exit any time, just type Bye!")
# while(flag==True):
#     user_response = input()
#     user_response=user_response.lower()
#     if(user_response!='bye'):
#         if(user_response=='thanks' or user_response=='thank you' ):
#             flag=False
#             print("BOT: You are welcome..")
#         else:
#             if(greet(user_response)!=None):
#                 print("BOT: "+greet(user_response))
#             else:
#                 sent_tokens.append(user_response)
#                 word_tokens=word_tokens+nltk.word_tokenize(user_response)
#                 final_words=list(set(word_tokens))
#                 print("BOT: ",end="")
#                 print(response(user_response))
#                 sent_tokens.remove(user_response)
#     else:
#         flag=False
#         print("BOT: Goodbye! Take care <3 ")

# from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder, OneHotEncoder, MinMaxScaler
# df = pd.read_csv("D://PythonPrgs/csvFiles/userprofile.csv")
# # from sklearn.datasets import mall
# # df =
# print(df.columns)
# print(df.describe())
# print(df.shape)
# print(df.isnull().sum())
# pd.set_option('display.max_columns', None)
# print(df.head())
# print(df['personality'].unique())
# for i in range(0, len(df['activity'])):
#     if df.loc[i, 'activity'] in ['professional', 'working-class']:
#         df.loc[i, 'class'] = "owner"
#     else:
#         df.loc[i, 'class'] = "consumer"
# # df.loc[5, 'c1'] = 'Value'
# # print(df.head())
# # print(df.tail())
# print(df['budget'].value_counts())
# catCols = df.select_dtypes(include=['object']).columns.tolist()
# print("Cat cols : \n", catCols)
# for i in catCols :
#     df[i] = LabelEncoder().fit_transform(df[i])
#     # df[i] = OneHotEncoder(categories=i).fit_transform(df[i]).toarray()
# print(df.head())
# df_std = StandardScaler().fit_transform(df)
# print(df_std)

# enc = preprocessing.OrdinalEncoder()
# >>> X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
# >>> enc.fit(X)
# OrdinalEncoder()
# >>> enc.transform([['female', 'from US', 'uses Safari']])
# print(len(df['buying']))
# # newcol = [i for i in range(0, 1728)]
# # df['alpha'] = newcol
# # print(df.columns)
# # print(df.head())