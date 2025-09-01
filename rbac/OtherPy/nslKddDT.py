import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# read the train and test dataset
train_data = pd.read_csv('D:/PythonPrgs/csvFiles/KddTrain.csv')
# test_data = pd.read_csv('D:/PythonPrgs/csvFiles/KDDTest21.csv')

# print(train_data.head())

# shape of the dataset
print('Shape of training data :',train_data.shape)
train_data = pd.DataFrame(train_data.values, columns=['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','class','prot'])
train_data['protocol_type'] = [0 if i == "tcp" else 1 if i == "udp" else 2 if i == "icmp" else 3 for i in train_data['protocol_type']]
print (train_data.head())
# train_data ['duration'] = train_data['duration'].astype(int)
train_x = train_data.drop(columns=['class'], axis=1)
print (train_x.head())
train_y = train_data['class']
print(train_y.head())
# train_x = train_data.drop(columns=['Survived'],axis=1)
# train_y = train_data['Survived']
print (train_data.info())
model = DecisionTreeClassifier()

# model.fit(train_x, train_y)
# print('success')
# Depth of the Decision Tree
# print('Depth of the Decision Tree : ', model.get_Depth())
