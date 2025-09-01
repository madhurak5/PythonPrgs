import pymongo
import PySimpleGUI as sg
sg.ChangeLookAndFeel("SystemDefaultForReal")
client = pymongo.MongoClient('mongodb://127.0.0.1/27017')  # when running Mongodb in the local server
welcome = [
        [sg.Text('Welcome to RBAC and ABAC Project', justification='center', font=("", 16), text_color="blue",size=(60, 2))],
        [sg.T("", size=(5, 1)), sg.Button("Admin", key='btnAdmin', size=(15, 1)),sg.T("", size=(5, 1)),
         sg.InputText("", key = 'fname', size=(15, 1)),sg.T("", size=(5, 1)), sg.InputText("", key = 'lname', size=(15, 1))]
         # sg.Button("Patient", key='btnPatient', size=(15, 1))],
        # [sg.T("",size=(55, 5)),sg.Button("Close", size=(10, 1), key='btnClose')]
    ]
welcomeWin = sg.Window("RBAC and ABAC project", layout=welcome, size=(650, 200),resizable=True)
    # tblCreate = 0
    # c = conn.cursor()
fname, lname ='', ''



while True:
    button, values = welcomeWin.read()
    if button == 'btnAdmin':
        fname = values['fname']
        lname = values['lname']
        print(fname, lname)
'''
mydb = client['Employee']
info = mydb.empInfo
ch = 'y'

while (ch == 'y'):
    nm = input("Your name :")
    lm = input("Your last name :")
    record = {
                'fname':nm,
                'lname':lm,
                # 'dept':'CS'
                }
            #     {
            #     'fname':'Mamta',
            #     'lname':'Mulimani',
            #     'dept':'CS'
            #     },
            #     {
            #     'fname':'Mahesh',
            #     'lname':'Mulimani',
            #     'dept':'CS'
            #     },
            # ]

    info.insert_one(record)
    ch = input("do u want to continue" )

print(info.find_one())
for rec in info.find({}):
    print(rec)
# ser = input("Enter name to search " )

for rec in info.find({'fname':{'$in':['Madhura', 'Mahesh']}}):
    print(rec)
print("----------------------")
for rec in info.find({'dept':'CS', 'fname':'Mahesh'}):
    print(rec)
print("**********************")
for rec in info.find({'$or':[{'dept':'CS'},{'fname':'Nidhi'}]}):
    print(rec)

info.update_many({'fname':'Madhura'}, {'$set':{'fname':'Madhur'}})
info.replace_one({"fname":"Nidhi"}, {"fname":"Nidhi", "lname":"Nandennavar", "dept":"EC"})'''

str1 = "AND$fname$lname$dept$OR$fname"
print(str1.split("$"))
acp = input("Enter the access control policy in the form {<logical gate>, attr1, attrb2 }")
acp1 = acp.split(",")

orGate = 0
andGate = 0
for i in range(len(acp1)):

    print(i, " ", len(acp1[i]), len(acp1[i].strip()))
    if acp1[i] == 'AND':
        andGate = andGate +1
    if acp1[i] == 'OR':
        orGate = orGate + 1
print("Or gates : ", orGate, "And Gates :", andGate)
rev_list = list(reversed(acp1))
print(rev_list)
import hashlib
myStr = "Hello"
hash_obj = hashlib.md5(myStr.encode())
print(hash_obj.hexdigest())
hash_obj = hashlib.sha1(b"Hello")
print(hash_obj.hexdigest())
hash_obj = hashlib.sha224(b"Hello")
print(hash_obj.hexdigest())
hash_obj = hashlib.sha256(b"Hello")
print(hash_obj.hexdigest())
print(hashlib.algorithms_available)
print(hashlib.algorithms_guaranteed)
hash_obj = hashlib.new('DSA')
hash_obj.update(b'Hello')
print(hash_obj.hexdigest())

import uuid

def hash_pwd(pwd):
    salt = uuid.uuid4().hex
    print(salt) #a7ca9d713ed9447e9d419bfafa9a85db
    return hashlib.sha256(salt.encode()+pwd.encode()).hexdigest()+":"+salt

def chk_pwd(hashed_pass, user_pwd):
    pass1, salt = hashed_pass.split(':')
    return pass1 == hashlib.sha256(salt.encode()+user_pwd.encode()).hexdigest()

new_pass = input("Enter a pwd : ")
hashed_pass1 = hash_pwd(new_pass)
print("String to store in db : " + hashed_pass1)
old_pass = input("Enter old pwd :" )
if chk_pwd(hashed_pass1,old_pass):
    print("Right pwd")
else:
    print("Pwd do not match")


__version__ = '1.5.0'
def ver_info():
    return tuple(map(int, __version__.split('.')))


print(ver_info())