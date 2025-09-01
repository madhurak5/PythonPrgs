import matplotlib.pyplot as plt
from scipy.stats import uniform
import numpy as np
import seaborn as sns
import pandas as pd



plt.grid(True)
#planets, diamonds, tips, exercise, flights
fName = "D:/PythonPrgs/titanic.csv"
db1 = pd.read_csv(fName)
attr = db1.keys()
myData1 = pd.DataFrame(db1, columns=attr)
print(myData1.head())
dbs = sns.load_dataset(db1, cache=True,data_home=None)
# sns.distplot(dbs['total_bill'], bins=40)
# sns.jointplot(x='passengers', y='year', data=dbs)
plt.show()





