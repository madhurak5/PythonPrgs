# # import numpy as np
# # import pandas as pd
# # from sklearn.model_selection import train_test_split
# # from sklearn.tree import DecisionTreeClassifier
# # from sklearn.preprocessing import LabelEncoder
# # from sklearn.preprocessing import MinMaxScaler, StandardScaler
# # data = pd.read_csv("D://PythonPrgs/rbacProj/credForm2.csv")
# # print(data.head())
# # print(data.describe())
# # print(data.dtypes)
# # cols = data.columns
# # print(cols)
# # # cols = ['Gender', 'Qualification', 'Designation', 'Experience', 'Department']
# # # df['DataFrame Column'].dtypes
# #
# # lab = LabelEncoder()
# # for i in cols:
# #     if data[i].dtypes == 'object':
# #         print(i)
# #         data[i] = lab.fit_transform(data[i])
# # print(data.head())
# #
# # data1 = data[['Gender','Qualification','Designation', 'Department', 'Role']]
# # print(data1)
# # X = data[['Gender','Qualification','Designation', 'Department']]
# # y = data['Role']
# #
# #
# # minMaxScaler = MinMaxScaler().fit(X)
# # X_scaler = minMaxScaler.transform(X)
# # X = X_scaler
# #
# # # X_std = StandardScaler().fit_transform(data)
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=41)
# # y_expect = y_test
# # print(X_train.shape)
# # print(y_train.shape)
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.naive_bayes import GaussianNB
# # from sklearn.tree import DecisionTreeClassifier
# # dtree = DecisionTreeClassifier()
# # print(dtree)
# # dtree.fit(X_train, y_train)
# # # ypred = dtree.predict(X_test)
# # # ypred = dtree.predict([[1,0,1,0]])
# # ypred = dtree.predict([[0,1,0,1]])
# # print("y Pred : ", ypred)
# # y1 = lab.inverse_transform(ypred)
# # print(y1)
# # ypred = dtree.predict([[1,0,1,0]])
# # print("y Pred : ", lab.inverse_transform(ypred))
# # ypred = dtree.predict([[0,1,2,2 ]])
# # print("y Pred : ", ypred)
# # y1 = lab.inverse_transform(ypred)
# # print(y1[0])
# # #
# #
# # #
# # #
# # X = data[['username','Fname','Gender','Experience','Qualification','Designation', 'Department']]
# # # X = data[[cols]]
# # y = data['Role']
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=41)
# # y_expect = y_test
# # print(X_train.shape)
# # print(y_train.shape)
# # from sklearn.feature_selection import SelectKBest, chi2, RFE
# # selFeatures = SelectKBest(chi2, k = 2).fit(X_train, y_train)
# # selFeatures_df = pd.DataFrame({'Feature' : list(X_train), 'Scores': selFeatures.scores_})
# # print(selFeatures_df)
# # selF = RFE(dtree, n_features_to_select=4, step=1)
# # selF = selF.fit(X, y)
# # print(selF.support_)
# #
# # print(selF .ranking_)
# # print(selF)
# #
# # a = "any1"
# # b = "AND"
# # c = "prof"
# # acp = [(a,b) ]
# #
# # # print(acp)
# # acp += [c]
# # # print(acp)
# # a = ['any', 'AND','Prof','OR','electronics', 'AND','Data Owner']
# # acp2 = [(a[0],a[1],a[2]) , a[0] ,(a[3], a[4], a[6])]
# # print("ACP " ,acp2)
# # # import itertools
# #
# # def chkList(i):
# #     if str(i).startswith("(") == True:
# #         print("\t\tA list")
# #         for j in i:
# #             print(j)
# #             if str(j).startswith("(") == True:
# #                 chkList(j)
# #
# # for i in acp2:
# #     print(i)
# #     chkList(i)
# #
# #
# # from owlready2 import *
# # onto_path.append("D://XACML")
# # onto = get_ontology("http://www.lesfleursdunormal.fr/static/_downloads/pizza_onto.owl")
# # onto.load()
# # class NonVegetarianPizza(onto.Pizza):
# #     equivalent_to = [
# #     onto.Pizza
# #     & ( onto.has_topping.some(onto.MeatTopping)
# #     | onto.has_topping.some(onto.FishTopping)
# #     ) ]
# #
# # test_pizza = onto.Pizza("test_pizza_owl_identifier")
# # test_pizza.has_topping = [ onto.CheeseTopping(),onto.TomatoTopping() ]
# # print(onto.Pizza)
# # print(test_pizza.has_topping.append(onto.MeatTopping()))
# # from ndg.saml.test import xml
# # from marshmallow import Schema, fields, post_load, ValidationError, validate
# # from typing import Union, List, Dict, TYPE_CHECKING
# # from py_abac import PDP, Policy, AccessRequest
# # import logging
# # rep= logging.getLogger()
# # rep.setLevel(logging.INFO)
# # rep.addHandler(logging.StreamHandler())
# # from sqlalchemy.orm import sessionmaker,scoped_session
# # from py_abac import Policy
# # from abc import ABCMeta, abstractmethod
# # from objectpath import Tree
# #
# # def is_sat(self, ctx)->bool:
# #     return any(value.is_sat(ctx) for value in self.values)
#
# #
# from pymongo import MongoClient
# # from py_abac import PDP, Policy, AccessRequest
# # from py_abac.storage import MongoStorage
# #
# # # Policy definition in JSON
# # policy_json = {
# #     "uid": "1",
# #     "description": "Max and Nina are allowed to create, delete, get any "
# #                    "resources only if the client IP matches.",
# #     "effect": "allow",
# #     "rules": {
# #         "subject": [{"$.name": {"condition": "Equals", "value": "Max"}},
# #                     {"$.name": {"condition": "Equals", "value": "Nina"}}],
# #         "resource": {"$.name": {"condition": "RegexMatch", "value": ".*"}},
# #         "action": [{"$.method": {"condition": "Equals", "value": "create"}},
# #                    {"$.method": {"condition": "Equals", "value": "delete"}},
# #                    {"$.method": {"condition": "Equals", "value": "get"}}],
# #         "context": {"$.ip": {"condition": "CIDR", "value": "127.0.0.1/32"}}
# #     },
# #     "targets": {},
# #     "priority": 0
# # }
# # # Parse JSON and create policy object
# # policy = Policy.from_json(policy_json)
# #
# # # Setup policy storage
# # client = MongoClient()
# # storage = MongoStorage(client)
# # # Add policy to storage
# # storage.add(policy)
# #
# # # Create policy decision point
# # pdp = PDP(storage)
# #
# # # A sample access request JSON
# # request_json = {
# #     "subject": {
# #         "id": "",
# #         "attributes": {"name": "Max"}
# #     },
# #     "resource": {
# #         "id": "",
# #         "attributes": {"name": "myrn:example.com:resource:123"}
# #     },
# #     "action": {
# #         "id": "",
# #         "attributes": {"method": "get"}
# #     },
# #     "context": {
# #         "ip": "127.0.0.1"
# #     }
# # }
# # # Parse JSON and create access request object
# # request = AccessRequest.from_json(request_json)
# #
# # # Check if access request is allowed. Evaluates to True since
# # # Max is allowed to get any resource when client IP matches.
# # assert pdp.is_allowed(request)
# #
#
# from py_abac import Policy
# policy_json = {
#     "uid":"1",
#     "description":"User Madhura is allowed to create, delete and get any file",
#     "effect":"allow",
#     "rules":{
#         "subject":{"$.name":{"condition":"Equals","value":"Madhura"}},
#         "resource":{"$.name":{"condition":"RegexMatch", "value":".*"}},
#         "action":[{"$.method":{"condition":"Equals", "value":"create"}},
#                   {"$.method":{"condition":"Equals", "value":"delete"}},
#                   {"$.method":{"condition":"Equals", "value":"get"}}],
#         "context":{}
#     },
#     "targets":{},
#     "priority":0
# }
# policy = Policy.from_json(policy_json)
# print(policy)
# #
#
#
# from py_abac import AccessRequest
# from flask import request, session
#
# request_json = {
#     "subject":{
#         "id":"",
#         "attributes":{"name":request.values.get("username")}
#     },
#     "resource":{
#         "id":"",
#         "attributes":{"name":request.path}
#     },
#     "action":{
#         "id":"",
#         "attributes":{"method":request.method}
#     },
#     "context":{}
# }
# request1 = AccessRequest.from_json(request_json)
# print(request1)
#
# # from pymongo import MongoClient
# # from py_abac import PDP
# # from py_abac.storage import MongoStorage
# # # def mgcl():
# # client = MongoClient()
# # db = client.test
# # # db.sites.insert
# # st = MongoStorage(client)
# # for p in policy:
# #     st.add(p)
# # pdp = PDP(st)
# # if pdp.is_allowed(request1):
# #     print("Access allowed")
# # else:
# #     print("Unauthorized access")
#
#
# # mgcl()
# from py_abac.policy.rules import Rules
# from py_abac import PDP, EvaluationAlgorithm
# from py_abac.storage import MongoStorage
# # from py_abac.providers import AttributeProvider
# # class EmailAttributeProvider(attributeProvider)
# #     def get_attribute_value(self, ace, attribute_path, ctx):
# #         return "example@gmail.com"
# # pdp = PDP(st,EvaluationAlgorithm.HIGHEST_PRIORITY,[EmailAttributeProvider])
# from py_abac.provider.base import AttributeProvider
#
# #
# from polynomial import Polynomial
# p = Polynomial (1,0, 3, 0)
# print(p)
# # print(p.derivative)
# print(p.nth_derivative())
# print(p.calculate(3))
# import numpy as np
# from scipy.interpolate import lagrange
# x = np.array([0,1,2])
# y = x ** 2
# poly = lagrange(x, y)
# print(poly)
# from numpy.polynomial.polynomial import Polynomial
# print(Polynomial(poly).coef)
# from PIL import Image, ImageDraw
# imgx, imgy = 800, 600
# image = Image.new("RGB", (imgx, imgy))
# draw = ImageDraw.Draw(image)

# from tate_bilinear_pairing import ecc
# import random
# a = random.randint(0, 10000)
# b = random.randint(0, 10000)
# g = ecc.gen()
# inf1, x1, y1 = ecc.scalar_mult(a, g)
# inf2, x2, y2 = ecc.scalar_mult(b, g)
# from tate_bilinear_pairing import eta
# t = eta.pairing(x1,y1,x2, y2)
# p1 = [inf1, x1, y1 ]
# # p2 = [inf2, x2, y2 ]
# # p3 = ecc.add(p1, p2)
# # import pbc
# # G = Bp
# # from ec import *
# # from ec import _getCachedValue, _equal, _serialize, _deserialize
# # from relic import librelic
# # from common import *
# # from ctypes import Structure, byref, sizeof, c_int, c_ulonglong
# #
# # from pbc import G1Element, G2Element, GtElement
# # g1 = G1Element()
# # print(g1)
#
# # from charm.toolbox.pairinggroup import PairingGroup, ZR,G1, G2, GT
# # import hashlib, binascii, os
# # salt = hashlib.sha256(os.urandom(60)).hexdigest().encode('ascii')
# # print(salt)
# # pwd = "madhura"
# # pwdhash = hashlib.pbkdf2_hmac('sha512',pwd.encode('utf-8'), salt,10000)
# # print(pwdhash)
# # pwdhash = binascii.hexlify(pwdhash)
# # print(pwdhash)
# # a = (salt+pwdhash).decode('ascii')
# # print(a)
#
# import json
# idVal = 9
# st = json.dumps([{'subject':{'id':str(idVal),'desc':'Policy'}},{'resource':'re'},'action',{'4':5,'6':7}],separators=(',',':'),sort_keys=False,indent=4)
# print(st)
# st2 = json.loads(st)
# print(st2)
# qual = 'B.E'
# g1 = 'and'
# dept = 'anyOf'
# g2 = 'or'
# desg = 'Asst. Professor'
# g3 = 'and'
# role = 'Data Consumer'
# if dept == 'anyOf':
#     dept = ['Electronics', 'Computer Science', 'Mechanical']
#     dpvals = "values"
# else:
#     dept = 'Electronics'
#     dpvals = "value"
# rsc = "hello.txt"
# # stw = json.dumps([{'subject':{'$department':dept}},'resource':'rsc','action':{'read', 'write'}}], separators=(',',':'),sort_keys=False, indent=4)
# # print(stw)
# #
# # {"uid": str(ui),"description": "Max and Nina are allowed to create, delete, get any resources only if the client IP matches.","effect": eff,"rules": {
# #         "subject": [{"$.department": {"condition": "Equals", dpvals: dept}},
# #                     {"$.qualification": {"condition": "Equals", qvals: quals}},
# #                     {"$.designation": {"condition": "Equals", dgvals: desgs}},
# #                     ],
# #         "resource": {"$.filename": {"condition": "RegexMatch", "value": rscval}},
# #         "action": [{"$.method": {"condition": "Equals", actvals: act1}},
# #                    {"$.method": {"condition": "Equals", actvals: act2}}],
# #         "context": {}  # "$.ip": {"condition": "CIDR", "value": "127.0.0.1/35"}
# #     },
# #     "targets": {},
# #     "priority": 0
# # }
#
# strtr = json.dumps([{"uid":'ui',"desc":"Description","rules":{"subject":[{"$.department": {"condition": "Equals", "value":dept}}, {"$.qualification": {"condition": "Equals", "value": "B.E"}}],"resrcs":{"filename":{"cond":"Reged","value":"helo"}},"action":{}}}],
#                    sort_keys=False, indent=4)
# print(strtr)
# pol = []
#
# st5 = {"uid":'ui',"desc":"Description","rules":{"subject":[{"$.department": {"condition": "Equals", "value":dept}}, {"$.qualification": {"condition": "Equals", "value": "B.E"}}],"resrcs":{"filename":{"cond":"Reged","value":"helo"}},"action":{}}}
# pol.append(st5) #= st5
# print(pol)

#
# import smtplib, ssl, getpass
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# port = 465
# sndMail = "madhurask375@gmail.com"
# recMail = "madhurasm@gmail.com"
# pwd = input("Password please :")
# # pwd = getpass.getpass()
# msg = MIMEMultipart("Test msg being sent using Python")
# msg['Subject'] = "Multipart Test"
# msg['From'] = sndMail
# msg['To'] = recMail
# text = """\
# Hi, How are you?
# Real Python has many great tutorials """
#
# html = """\
# <html>
#     <body>
#         <p>  Hi, <br>
#         How are you?<br>
#         </p>
#     </body>
# </html>
# """
# part1 = MIMEText(text, "plain")
# part2 = MIMEText(html, "html")
# msg.attach(part1)
# msg.attach(part2)

# ctx = ssl.create_default_context()
# server1 = smtplib.SMTP("smtp.gmail.com", 587)
# server1.starttls()
# server1.login(sndMail,pwd)
# print("login success ")
# server1.sendmail(sndMail, recMail, msg.as_string())
# print("Email sent to ", recMail)

# with smtplib.SMTP_SSL("smtp.gmail.com", port, context=ctx) as server:
#     server.login("madhurask375@gmail.com", pwd)
# from charm.toolbox.integergroup import IntegerGroup
# # from charm.toolbox.pairinggroup import G1
# import charm.toolbox.pairinggroup
# grp1 = IntegerGroup()
# g = grp1.paramgen(1024)
# grp2 =
# import javaobj
# with open("D://JavaPrgs/cpabe/masterkey.txt", "rb") as fd:
#     jobj =fd.read()
# pobj =javaobj.loads(jobj)
# print(pobj)
#
# from secretsharing import SecretSharer
# from secretsharing import PlaintextToHexSecretSharer
# # shares = SecretSharer.split_secret("345939485898", 2, 3)
# shares = PlaintextToHexSecretSharer.split_secret("correct horse battery staple", 2, 3)
# print(shares)
#
# import random
# from math import ceil
# from decimal import *
#
# global field_size
# field_size = 10 ** 5
#
#
# def reconstructSecret(shares):
#     # Combines shares using
#     # Lagranges interpolation.
#     # Shares is an array of shares
#     # being combined
#     sums, prod_arr = 0, []
#
#     for j in range(len(shares)):
#         xj, yj = shares[j][0], shares[j][1]
#         prod = Decimal(1)
#
#         for i in range(len(shares)):
#             xi = shares[i][0]
#             if i != j: prod *= Decimal(Decimal(xi) / (xi - xj))
#
#         prod *= yj
#         sums += Decimal(prod)
#
#     return int(round(Decimal(sums), 0))
#
#
# def polynom(x, coeff):
#     # Evaluates a polynomial in x
#     # with coeff being the coefficient
#     # list
#     return sum([x ** (len(coeff) - i - 1) * coeff[i] for i in range(len(coeff))])
#
#
# def coeff(t, secret):
#     # Randomly generate a coefficient
#     # array for a polynomial with
#     # degree t-1 whose constant = secret'''
#     coeff = [random.randrange(0, field_size) for _ in range(t - 1)]
#     coeff.append(secret)
#
#     return coeff
#
#
# def generateShares(n, m, secret):
#     # Split secret using SSS into
#     # n shares with threshold m
#     cfs = coeff(m, secret)
#     shares = []
#
#     for i in range(1, n + 1):
#         r = random.randrange(1, field_size)
#         shares.append([r, polynom(r, cfs)])
#
#     return shares
#
#
# # Driver code
# if __name__ == '__main__':
#     # (3,5) sharing scheme
#     t, n = 3, 7
#     secret = 348349
#     print('Original Secret:', secret)
#
#     # Phase I: Generation of shares
#     shares = generateShares(n, t, secret)
#     print('\nShares:', *shares)
#
#     # Phase II: Secret Reconstruction
#     # Picking t shares randomly for
#     # reconstruction
#     pool = random.sample(shares, t)
#     print('\nCombining shares:', *pool)
#     print("Reconstructed secret:", reconstructSecret(pool))


# import glob
# print(glob.glob("D:/0Research"))
# # for file_name in glob.iglob('D:/0Research/0Pdfs/3-2017/2Feb17', recursive=True):
# #   print(file_name)
#
# import os
#
# arr = os.listdir('D:/0Research/0Pdfs/2-2018/12Dec18')
# # arr = os.listdir('D:/0Research/0Pdfs/2-2018/9Sep18') 1Jan 2Feb 3Mar 4Apr 5May 6Jun 7Jul 8Aug 9Sep 10Oct 11Nov 12Dec
# for i in arr:
#     print(i)
# import PyPDF2
# pdf_file = open("D:/0Research/0Pdfs/0-2020/0Aug20/1A Homomorphic Universal Re-encryptor for Identity Based Encryption 2 Aug 20.pdf") #D:\0Research\0Pdfs\0-2020\0Aug20
# read_pdf = PyPDF2.PdfFileReader(pdf_file)
# number_of_pages = read_pdf.getNumPages()
# page = read_pdf.getPage(0)
# page_content = page.extractText()
# print (page_content)
# import textract
# # text = textract.process("D:/0Research/0Pdfs/0-2020/0Aug20/1A Homomorphic Universal Re-encryptor for Identity Based Encryption 2 Aug 20.pdf")
# # print(text)
# # from tika import parser
# # # import parser
# # raw = parser.from_file("D:/0Research/0Pdfs/0-2020/0Aug20/1A Homomorphic Universal Re-encryptor for Identity Based Encryption 2 Aug 20.pdf")
# # raw = str(raw)
# # safe_text = raw.encode('utf-8', errors='ignore')
# #
# # safe_text = str(safe_text).replace("\n", "").replace("\\", "")
# # print('--- safe text ---' )
# # print( safe_text )
# #
# # import PyPDF2
# # pdfFileObject = open("D:/0Research/0Pdfs/0-2020/0Aug20/1A Homomorphic Universal Re-encryptor for Identity Based Encryption 2 Aug 20.pdf", 'rb')
# # pdfReader = PyPDF2.PdfFileReader(pdfFileObject)
# # count = pdfReader.numPages
# # print(count)
# # print(pdfReader.getPage(1).extractText())
# # print()
#
# # import zlib
# # import base64
# # import rsa
# # privKey, pubKey = rsa.newkeys(512)
# # # cip = rsa.encrypt(b'Hello World!!!',privKey)
# # cip = rsa.encrypt(b'Hello World!!!',pubKey) # Right usage
# # base64Text = base64.b64encode(cip).decode()
# # print("Cipher: ",cip)
# # print("Base64Text: ",base64Text)
# # # text = rsa.decrypt(base64.b64decode(base64Text.encode()),pubKey)
# # text = rsa.decrypt(base64.b64decode(base64Text.encode()),privKey)  # Right usage
# # print("After decoding: ",text.decode())
# #
# # # import PK
#
# import tensorflow as tf
# tf.keras.models
# #
# # h = hash("h") # h - 5000641251790643283
# # print(h)
# # import zlib
# # str1 = "hello"
# # p = zlib.adler32(b"hello")
# # print(p)
# # p = zlib.crc32(b"hello")
# # print(p)
# #
# # import hashlib
# # # hashlib.md5(b'hello')
# # print(hashlib.sha256(b"hello").hexdigest())
# # print(len(hashlib.sha256(b"hello").digest()))
# # from PIL import Image
# # from binascii import hexlify
# # data = 'Sending encrypted'
# # data = data.encode('utf-8')
# # sha3_512 = hashlib.sha3_512(data)
# # sha3_512_digest = hashlib.sha3_512(data).digest()
# # sha3_512_hexDigest = hashlib.sha3_512(data).hexdigest()
# # print("Digest Output: ", sha3_512_digest)
# # print("Hex output : ", sha3_512_hexDigest)
# # print("Binary Hex Output: ", hexlify(sha3_512_digest))
# # from hashlib import sha256
# # from secrets import compare_digest
# # sha256_digest1 = sha256(b'Hello world')
# # digest1 =  sha256_digest1.digest()
# # hexDigest1 = sha256_digest1.hexdigest()
# # sha256_digest2 = sha256()
# # sha256_digest2.update(b'Hello')
# # sha256_digest2.update(b' world')
# # digest2 =  sha256_digest2.digest()
# # hexDigest2 = sha256_digest2.hexdigest()
# # print(compare_digest(digest1,digest2))
# # print(compare_digest(hexDigest1,hexDigest2))
# #
# # from hashlib import blake2b
# # data = b'Msg for transmission'
# # blake = blake2b(data,digest_size=32)
# # print("Blake Digest : ", blake.digest())
# # print("Blake Hex Digest : ", blake.hexdigest())
# # from Cryptodome.Hash import Poly1305
# # from Cryptodome.Cipher import AES
# # key = b'The key size has to be 32 bytes!'
# # mac = Poly1305.new(key=key, cipher=AES)
# # mac.update(b'Msg to be delivered')
# # mac_nonce= mac.nonce
# # mac_hexDigest = mac.hexdigest()
# # print('Poly1305 nonce: ', mac_nonce)
# # print('Poly1305 Hex Digest: ', mac_hexDigest)
# # import hashlib
# # print("Algorithms: ", end="")
# # print(hashlib.algorithms_available)
# # print(hashlib.algorithms_guaranteed)
# # #
# # # import requests
# # # proxies = {
# # #     "http": "http://10.10.10.10:8000",
# # #     "https": "http://10.10.10.10:8000",
# # # }
# # # r = requests.get("http://gmail.com", proxies=proxies)
# # # print(r)
# # from tate_bilinear_pairing import eta
# # eta.init(151)
# # from tate_bilinear_pairing import ecc
# # g = ecc.gen()
# # import random
# # a = random.randint(0,1000)
# # b = random.randint(0,1000)
# # print("g: ",g)
# # inf1, x1, y1 = ecc.scalar_mult(a, g)
# # inf2, x2, y2 = ecc.scalar_mult(b, g)
# # print("inf1: ", inf1)
# # print("x1: ", x1)
# # print("y1: ", y1)
# # print("inf2: ", inf2)
# # print("x2: ", x2)
# # print("y2: ", y2)
# # t = eta.pairing(x1, y1, x2, y2)
# # print('t = ', t)
# # import django
# # from django.http import HttpResponse
# # from django.shortcuts import render
# # from django.contrib.auth.decorators import login_required
# # from django.core.wsgi import get_wsgi_application
# # import os
# # os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Demo.Settings")
# # appln = get_wsgi_application()
# # from xml.dom import minidom
# # mydoc = minidom.parse('D:\PythonPrgs\\rbacProj\\venv\items.xml')
# # items = mydoc.getElementsByTagName('item')
# # print("Item #2 attrib : ")
# # print(items[1].attributes['name'].value)
# # print("All Attribs")
# # flg = 0
# # for elem in items:
# #     chkElem = elem.attributes['name'].value
# #     if chkElem == "item9":
# #         flg = 1
# #     else:
# #         flg = 0
# # if flg == 1:
# #     print("found ur elem")
# # else:
# #     print("Sorry! dint find ur elem")
# #
# # print('\nItem #2 data:')
# # # print(items[1].data)
# # print(items[1].childNodes[0].data)
# # import xmlsec
# # from lxml import etree
# # mgr = xmlsec.KeysManager()
# # # key = xmlsec.Key.from_file('rsakey.pem', xmlsec.constants.KeyDataFormatPem)
# # # mgr.add_key(key)
# # enc_ctx = xmlsec.EncryptionContext(mgr)
# # print(enc_ctx)
# # root = etree.parse("D:\PythonPrgs\\rbacProj\\venv\items.xml").getroot()
# # print(root)
#
# # a = {"Type":"A", "f1":"v1", "f2":"v2","f3":"v3"}
# # print(repr(a))
# # print(a)
# # with open("D:/PythonPrgs/file1.py","r+") as f:f.write(repr(a))
#
def printListofFiles(getfilesfrom):
    import os

    arr = os.listdir(getfilesfrom) #D:\0Research\0Pdfs\0-2020\9Sep20 D:\0Research\0Pdfs\2015\Cloud Papers D:\0Research\0Pdfs\0-2020\IEEE Access Control Papers
    # arr = os.listdir('D:/0Research/0Pdfs/2-2018/9Sep18') 1Jan 2Feb 3Mar 4Apr 5May 6Jun 7Jul 8Aug 9Sep 10Oct 11Nov 12Dec
    for i in arr:
        print(i)


#
#
# # printListofFiles()
# class person:
#     id = 1
#     def initVar(self, id):
#         self.id = id
#         self.name = input("Enter the name ")
#         self.age = input("Enter the age : ")
#         self.role = input("Enter the role : ")
#         return self
#     def prnDetails(self):
#         print(self.id, self.name, self.age, self.role)
#
# p = person()
# p.initVar(10)
# p.prnDetails()
#
# class base1:
#     pass
# class der1(base1):
#     pass
# class der2 (der1):
#     pass
#
# print(der2.__mro__)
# print(der2.mro())
# class X:
#     pass
# class Y:
#     pass
# class Z:
#     pass
# class A(X, Y):
#     pass
# class B(Y, Z):
#     pass
# class M(A, B, Z):
#     def __init__(self, x = "m"):
#         self.x = x
#     def __str__(self):
#         return ("{0}".format(self.x))
# print(M.__mro__)
# # print(q)
# m = M("A")
# print(m)
# import itertools
#
# def slicefile(filename, start, end):
#     lines = open(filename)
#     return itertools.islice(lines, start, end)
#
# out = open("D:\\0Research\\0Pdfs\\2015\\hello1.txt", "w")
# for line in slicefile("D:\\0Research\\0Pdfs\\2015\\E.pdf", 0, 4):
#     out.write(line)

# Efficient Revocation in ciphertext-policy attribute based encryption based cryptographic storage

import os


# fil = "D:\\0Research\\0Pdfs\\2015\\E.pdf"
# outfil = "D:\\0Research\\0Pdfs\\2015\\hello1.txt"
#
# f = open(fil,'r')
#
# numbits = 1000000000
#
# for i in range(0,os.stat(fil).st_size/numbits+1):
#     o = open(outfil+str(i),'w')
#     segment = f.readlines(numbits)
#     for c in range(0,len(segment)):
#         o.write(segment[c]+"\n")
#     o.close()

# from PyPDF2 import PdfFileWriter, PdfFileReader
#
# inputpdf = PdfFileReader(open("D:\\0Research\\0Pdfs\\2015\\E.pdf", "rb"))
#
# for i in range(inputpdf.numPages):
#     output = PdfFileWriter()
#     output.addPage(inputpdf.getPage(i))
#     with open("D:\\0Research\\0Pdfs\\2015\\E%s.pdf" % i, "wb") as outputStream:
#         output.write(outputStream)
# class User:
#     name = ""
#     age = 0
#     def prnUserDetails(self):
#         print("Printing user details")
#         print(self.name)
#         print(self.age)
#
#     def getUserDetails(self):
#         self.name = input("Enter the User's name : ")
#         self.age = input("Enter the User's age : ")
# # u = []
# u1 = User()
# n = input("Enter total no of users ")
# for i in range (0, int(n)):
#     u1.getUserDetails()
#     print(u1.prnUserDetails())
class Admin:
    def createUser(self):
        # pass
        print("Creating user")

    def delUser(self):
        # pass
        print("Deleting user")

    def createRole(self):
        pass

    def delRole(self):
        pass

    def assignPerm(self):
        pass

    def deassignPerm(self):
        pass

    def assignRole(self):
        pass

    def deassignRole(self):
        pass

#
print("Menu")
print("1. Create User")
print("2. Delete User")
print("3. Create Role")
print("4. Delete Role")
print("5. Assign Permissions -> Role")
print("6. Deassign Permissions -> Role")
print("7. Assign Role -> User")
print("8. Deassign Role -> User")
print("9. Exit")
a = Admin()
choice = "" #int(input("Enter ur choice: "))
if choice == 1:
    a.createUser()
elif choice == 2:
    a.delUser()
elif choice == 3:
    a.createRole()
elif choice == 4:
    a.delRole()
elif choice == 5:
    a.assignPerm()
elif choice == 6:
    a.deassignPerm()
elif choice == 7:
    a.assignRole()
elif choice == 8:
    a.deassignRole()
else:
    print("Invalid choice")
import os
# op = subprocess.call()

# printListofFiles()
from tokenize import tokenize, untokenize, NUMBER, STRING, NAME, OP
from io import BytesIO

# return untokenize(result).decode('utf-8')


# def decistmt(s):
#     result = []
#     g = tokenize(BytesIO(s.encode('utf-8')).readline)  # tokenize the string
#     for toknum, tokval, _, _, _ in g:
#         if toknum == NUMBER and '.' in tokval:  # replace NUMBER tokens
#             result.extend([
#                 (NAME, 'Decimal'),
#                 (OP, '('),
#                 (STRING, repr(tokval)),
#                 (OP, ')')
#             ])
#         else:
#             result.append((toknum, tokval))
#     return untokenize(result).decode('utf-8')
#
# st = decistmt("Hello World!")
# print("Res: ",st)
# def user():
#     n = input("Enter ur name: ")
#     a = getAccess(n)
#     if a:
#         print("Access granted")
#     else:
#         print("no access")
#
# def getAccess(i):
#     if i == 'Madhura':
#         g = 1
#     else:
#         g = 0
#     return g
#
# user()
# from crypto.Util.number import *
# https://asecuritysite.com/encryption/rsa12
# from Cryptodome.Util.number import bytes_to_long, long_to_bytes
# from Cryptodome.Random import get_random_bytes
# import Cryptodome
# import libnum
# import sys
#
# bits=60
# msg="madhura hello"
#
# if (len(sys.argv)>1):
#         msg=str(sys.argv[1])
# if (len(sys.argv)>2):
#         bits=int(sys.argv[2])
#
# p = Cryptodome.Util.number.getPrime(bits, randfunc=get_random_bytes)
# q = Cryptodome.Util.number.getPrime(bits, randfunc=get_random_bytes)
#
#
#
# n = p*q
# PHI=(p-1)*(q-1)
#
# e=65537
# d=libnum.invmod(e,PHI)
#
# m=  bytes_to_long(msg.encode('utf-8'))
#
# c=pow(m,e, n)
# res=pow(c,d ,n)
#
# print ("Message=%s\np=%s\nq=%s\n\nd=%d\ne=%d\nN=%s\n\nPrivate key (d,n)\nPublic key (e,n)\n\ncipher=%s\ndecipher=%s" % (msg,p,q,d,e,n,c,(long_to_bytes(res))))

import policies

import rbac.acl

acl = rbac.acl.Registry()
fil2 = 'D:/0Research/0Pdfs/0-2020/IEEE Revocation Papers'
acl.add_role("doctor")
# acl.add_role("nurse", "doctor")
# acl.add_resource("patient details")
acl.add_resource(fil2)
# # # r1 = "nurse"
# flag = 0
# acl.add_role("patient", ["nurse","doctor"])
# acl.allow("nurse", "view", "patient")
# acl.allow("nurse", "view", "patient details", None)
acl.allow("doctor", "view", fil2, None)
# acl.deny("nurse", "edit", "patient details")
if acl.is_allowed("doctor","view", fil2):
    print("Nurse can edit the patient details")
    flag = 1
else:
    print("Nurse not allowed to edit patient details")

# if flag == 1:
#     print("A vaild member is accessing")
# else:
#     print("unauthozd access ")

# def adminRbac():
# policies.

import vakt
from vakt.rules import Eq, Any, StartsWith, And, Greater, Less
policy = vakt.Policy(
    123456,
    actions=[Eq('fork'), Eq('clone')],
    resources=[StartsWith('repos/Google', ci=True)],
    subjects=[{'name': Any(), 'stars': And(Greater(50), Less(999))}],
    effect=vakt.ALLOW_ACCESS,
    context={'referer': Eq('https://github.com')},
    description="""
    Allow to fork or clone any Google repository for
    users that have > 50 and < 999 stars and came from Github
    """
)
storage = vakt.MemoryStorage()
storage.add(policy)
guard = vakt.Guard(storage, vakt.RulesChecker())

inq = vakt.Inquiry(action='fork',
                   resource='repos/google/tensorflow',
                   subject={'name': 'larry', 'stars': 80},
                   context={'referer': 'https://github.com'})

assert guard.is_allowed(inq)
from easyrbac import Role, User, AccessControlList
r = "everyone"
ra = "admin"
everyone_role = Role(r)
admin_role = Role(ra)
e_user = User(roles=[everyone_role])
a_user = User(roles=[admin_role, everyone_role])
acl = AccessControlList()
print(e_user)
print(a_user)
# acl.

from cryptography.fernet import Fernet
key = Fernet.generate_key()
f = Fernet(key)
token =  f.encrypt(b"A really secret message. Not for prying eyes.")
print("TimeStamp: " ,Fernet.extract_timestamp(f,token))  # 1607969957
print(Fernet.mro())  # [<class 'cryptography.fernet.Fernet'>, <class 'object'>]
print("Token : ",token) #  b'gAAAAABf16yl1F9iY8ANUx9y4fyWSNddEhBpKYhd6Gbkbq2vxw2LzbRG15aclPanalxf6r9Z7E3jmV2Hq-imS7r4v2I93sBgP63nAmDTPNyD0PA2WJS2RyZw3By0L3zrL7ALdFZVhUW7'
print("DEcryption : ",f.decrypt(token))
print(len(token)) # 140
# from cryptography.hazmat.primitives.twofactor.hotp import HOTP
import secrets
import string
alphabet = string.ascii_letters + string.digits
password = ''.join(secrets.choice(alphabet) for i in range(10))
print(password) #YNxLJXhvGF
import datetime;
ts = datetime.datetime.now().timestamp()
print("Timestamp : ",ts)
ts1 = ts + 1000
print("Incremented ts : ", ts1)
dif = ts1 - ts
print("Dif : ", dif)
import time
import datetime
start = time.time()
readable = datetime.datetime.fromtimestamp(ts).isoformat()
print("Readable timestamp: ", readable)

import time
ts = time.gmtime()
print(time.strftime("%Y-%m-%d %H:%M:%S", ts))
# 2020-12-01 16:12:36

print(time.strftime("%x %X", ts))
# 12/01/20 16:12:36

# Iso Format
print(time.strftime("%c", ts))
# Tue Dec  1 16:12:36 2020

# Unix timestamp
# print(time.strftime("%s",ts))
# 1606835556
ts1 = time.gmtime()
import datetime
from datetime import timedelta

datetimeFormat = '%Y-%m-%d %H:%M:%S.%f'
date1 = '2016-03-10 10:00:28.585'
date3 = time.gmtime()
print("Cur time : ", date3)

date2 = '2016-03-10 09:56:28.067'
diff = datetime.datetime.strptime(date1, datetimeFormat) \
       - datetime.datetime.strptime(date2, datetimeFormat)

print("Difference:", diff)
print("Days:", diff.days)

# print("Microseconds:", diff.microseconds)
# mins = diff.seconds/60
# print("Minutes:", mins)
# if mins >= 5:
#     print("token expired")
# else:
#     print("Token valid for another ")
# Q = "anyof"
# qua = ["MBBS", "MS", "MD"]
# # print(qua)
# # for i in qua:
# #     print(i)
# q = "ME"
# if Q == "anyof":
#     if q in qua:
#         print("allowed")
#     else:
#         print("No")
# else:
#     print("A specific value is reqd")
# D = "anyof"
# dep = ["Radiology", "Cardiology", "Oncology"]
# d = "Radiology"
# # acp = "(Q or D) and (P and (Q or K) or (A and B))"
# kws = {'AND', 'OR'}
# attribs = {'Designation', 'B', 'C', 'D', 'E'}
# acp = "(Designation='AP' AND B='Pres' OR C) AND(D OR E OR F)"
# tok = acp.split(")")
# lst = []
# i = 0
# import re
# # s = 'Name(something)'
# res = re.search('\(([^)]*)', acp).group(1)
# # 'something'
# print("Single search : ", res)
#
# res1 = re.findall('\(([^)]*)', acp)
# print("Trying with * :",res1)
# print("hello")
# k = ""
# l = ""
# for i in res1:
#     j = i.split(" ")
#     for k in j:
#         if k in attribs:
#             print("found attrib : ", k)
#         if k not in kws:
#             if k.find("=") != -1:
#                 l = k.split("=")
#                 for m in range (0, len(l)):
#                     if l[m] in attribs:
#                         print("attrib found", l[m])
#                     else:
#                         print("attrib value", l[m])
#             else:
#                 print(k)
# flag = 0
# if Q == "anyof" or D == "anyof":
#     if q in qua or d in dep:
#         print("allowed")
#         flag = 0
#     else:
#         print("NOt allowed")
#         flag = 1
#



# import uuid
# # Printing random id using uuid1()
# print ("The random id using uuid1() is : ",end="")
# print (uuid.uuid1())
# import pandas as pd
# a = [1, 2,-4,  5]
# c = pd.Series(a,None)
# print(c>1)
# c = c + 1

# print( == 1)
# from opencensus.ext.stackdriver import trace_exporter as stackdriver_exporter
# import opencensus.trace.tracer
#
#
# def initialize_tracer(project_id):
#     exporter = stackdriver_exporter.StackdriverExporter(
#         project_id=project_id
#     )
#     tracer = opencensus.trace.tracer.Tracer(
#         exporter=exporter,
#         sampler=opencensus.trace.tracer.samplers.AlwaysOnSampler()
#     )
#
#     return tracer
# GOOGLE_CLOUD_PROJECT = "majestic-trail-299301"
# t = initialize_tracer(GOOGLE_CLOUD_PROJECT)
# print(t)
# import hashlib
# import hmac
# update_bytes = b'hello world!!!'
# # update_bytes = open("D://PythonPrgs/Simple.txt")
# pwd = b'402xy5#'
# my_hmac = hmac.new(pwd, update_bytes, digestmod=hashlib.sha1)
# print("First digest : ", str(my_hmac.digest()))
#
# print("Canonical name : ", my_hmac.name)
# print("Block size : ", str(my_hmac.block_size) + " bytes")
# print("Digest size : ", str(my_hmac.digest_size) + " bytes")
# my_hmac_copy = my_hmac.copy()
# print("Copied hmac : ", str(my_hmac_copy.digest()))
# First digest :  b'\xedE\x1e)\xda\'"\x1f#\xf1\xa6\xb4\x86\xf3\x15\xb9'
# Canonical name :  hmac-md5
# Block size :  64 bytes
# Digest size :  16 bytes
# Copied hmac :  b'\xedE\x1e)\xda\'"\x1f#\xf1\xa6\xb4\x86\xf3\x15\xb9'
print('hello madhura')
# D:\0Research\0Pdfs\My work\2017\2Feb17 D:\0Research\0Pdfs\2015\Cloud Papers\1CC Sec Issues D:\0Research\0Pdfs\1-2019\1Jan19
# D:\0Research\0Pdfs\0-2021\IEEE papers D:\0Research\0Pdfs\2-2018\12Dec18
# D:\0Research\0Pdfs\0-2020
printListofFiles('D:/0Research/0Pdfs/0-2020/IEEE Accountability') #D:\0Research\0Pdfs\0-2021\1Jan21 D:\0Research\0Pdfs\2015\Cloud Papers\1CC Sec Issues
# D:\0Research\0Pdfs\1-2019\11Nov19
# 1CC Sec Issues 2Access Control 5User Revocation
# D:\0Research\0Pdfs\0-2021\3Mar21\Book Chapter 12 Mar 21

# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
# df = pd.read_csv("D://Pythonprgs/csvFiles/wdbc.data") # metaData.csv
# print("Wdbc data")
# print(df.info())
# print(df.shape)
# print("Wdbc data rows")
# print(df.head())
# cols = df.columns
# print(cols)
# catCols = df.select_dtypes(include=['object']).dtypes
# print("Cat Cols \n", catCols)
#
# corrMat = df.corr()
# print(corrMat)
# sns.heatmap(corrMat, annot=True)
# # plt.show()
# ohe = OneHotEncoder()
# le = LabelEncoder()
# le_Date = LabelEncoder()
# ohe_date = OneHotEncoder()
# le_State = LabelEncoder()
# le_country = LabelEncoder()
# le_Update = LabelEncoder()
# print(df['ObservationDate'])
# # df['Province'] = df['Province/State']
# print(df['Province/State'].head(100))
# df['Province/State'] = df['Province/State'].fillna(0)
# print(df['Province/State'].head(100))
# # df['ObservationDate'] = le_Date.fit_transform(df['ObservationDate'])
# df['Province/State'] = le_State.fit_transform(df['Province/State'])
# # df['Country/Region'] = le_country.fit_transform(df['Country/Region'])
# # df['Last Update'] = le_Update.fit_transform(df['Last Update'])
# print(df['Province/State'].head(100))
# print(df['ObservationDate'].value_counts())
# for i in range (0, 10):
#     print(i)
#     for j in range(0, len(cols)):
#         print(df.iloc[i, j])

# print(df['Id'].unique())
# today1 = df[df.TotalSteps > 1000]
# print(today1)
# toda1 = df[df.ActivityDate == '4/21/2016']
# print(toda1)
# dataset.to_csv("D:/PythonPrgs/csvFiles/Automobile_df.csv")
# active = df['Confirmed'] - df['Deaths'] - df['Recovered']
# top = df[df['Last_Update'] == df['Last_Update'].min()]
# world = top.groupby('Country_Region')['Confirmed','Deaths','Recovered'].sum().reset_index()
# print(world)
# print(top)
# print(active, df['Active'])
import plotly
# import plotly.express as px
# import plotly.graph_objects as go
# import seaborn as sns
# figure = px.choropleth(world, locations="Recovered")
# figure.show()
#
# from sklearn.feature_extraction.text import CountVectorizer
# feedback = []
# for i in df['ObservationDate'].values:
#     # print(i)
#     date9 = " ".join(e for e in i.split())
#     feedback.append(date9.lower().strip())
#
# print(feedback)

# import pandas as pd
# df = pd.read_csv("http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv",index_col=0)
# print(df.head())
import matplotlib.pyplot as plt
plt.show()
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
# from tensorflow.keras.applications.vgg19 import VGG19

from keras.applications.vgg19 import VGG19
model = VGG19(weights='imagenet')
# model = load_model()

print(model.summary())
model.save('vgg19.h5')
app = Flask(__name__)
model_path = 'vgg19.h5'
model1 = load_model(model_path)

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils