
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from copy import deepcopy

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
plt.show()

data = pd.read_csv("D:/PythonPrgs/xclara.csv")
print(data.head())
print(data.shape)
print (data)
f1 = data['V1'].values
f2 = data['V2'].values
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='black', s=7)
plt.plot(show=True)
