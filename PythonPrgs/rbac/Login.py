# import
import sqlite3
import PySimpleGUI as sg
import chooseRoles
sg.ChangeLookAndFeel("SystemDefaultForReal")
import pymysql.connections
import mysql.connector
from Abac_DC import dataConsumerForm
from Abac_DO import dataOwnerForm
conn = sqlite3.connect('examplemm.db')
c = conn.cursor()

def login(typeofUser):
    loginLay = [
                [sg.Text('Login Form for Existing Users', size=(60, 2), justification='center', font=("", 16),text_color='blue')],
                [sg.Text('Username', size=(20, 1)), sg.InputText('',key='username',focus=True, size=(30, 1))],
                [sg.Text("Password", size=(20, 1)), sg.In(key='password', password_char='*', size=(30, 1))],
                [sg.T(" ", size=(20,1))],
                [sg.T(" ", size=(20,1)),sg.Button("Login", key='btnLogin', size=(10,1)), sg.Button("Reset", key='btnReset', size=(10,1))],
                [sg.T(" ", size=(20, 1))],
                [sg.T(" ", size=(50, 1)),sg.Button("Logout", key='btnExit', size=(10,1))]
     ]
    loginWin = sg.Window("Login to Rbac project", layout=loginLay, resizable=True,size=(600, 300))
    flag = 0

    while True:
        button, values = loginWin.read()
        if button is None or button == 'btnExit':
            break

        if button == 'btnLogin':
            uname = values['username']
            pwd = values['password']
            # selRole = values['roleCombo']
            if typeofUser == 'staff':
                recstaff = c.execute("select username, Pwd, Role from Staff where username = ? and Pwd = ? and AARequest = 0" ,(uname, pwd))
                recstaff1 = recstaff.fetchall()
                if recstaff1 == '' or recstaff1 == []:
                    print("No records")
                    flag = 1
                    break
                else:
                    print(recstaff1[0][0], recstaff1[0][1], recstaff1[0][2])
                    chooseRoles.chooseTheRole(uname)
                    # role = recstaff1[0][2]
                    # if role == "Data Consumer":
                    #     dataConsumerForm(uname)
                    # else:
                    #     dataOwnerForm(uname)
            elif typeofUser == 'patient':
                recpatient = c.execute("select username, Pwd, Role from patient where username = ? and Pwd = ? and AARequest = 0 ", (uname, pwd))
                recpatient1 = recpatient.fetchall()
                if recpatient1 == '' or recpatient1 == []:
                    print("No records")
                    flag = 1
                    break
                else:
                    print(recpatient1[0][0], recpatient1[0][1], recpatient1[0][2])
                    role = recpatient1[0][2]
                    chooseRoles.chooseTheRole(uname)
                    # if role == "Data Consumer":
                    #     dataConsumerForm(uname)
                    # else:
                    #     dataOwnerForm(uname)
            elif typeofUser == 'admin':
                recadmin = c.execute("select username, Pwd from credForm2 where username = ? and Pwd = ?", (uname, pwd))
                recadmin1 = recadmin.fetchall()
                if recadmin1 == '' or recadmin1 == []:
                    print("No records")
                    flag = 1
                    break
                else:
                    print(recadmin1[0][0], recadmin1[0][1])

        elif button == 'btnReset':
            elems = ['username','password']
            for i in elems:
                loginWin.FindElement(i).Update('')
            loginWin.FindElement('username').set_focus()
        elif button == 'btnExit':
            loginWin.Close()
    loginWin.Close()
# login('patient')