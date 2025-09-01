import requests
url = 'http://localhost:12345/predict_api'
r = requests.post(url, json={'sepal length':5.1, 'sepal width':3.5, 'petal length':1.4, 'petal width':0.2})
print(r.json())