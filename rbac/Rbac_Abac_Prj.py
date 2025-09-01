from __future__ import print_function
from mysql.connector import errorcode
import sqlite3
import os
import pymysql
import PySimpleGUI as sg
import pymysql.connections
import mysql.connector
from Abac_admin import admin
from Abac_DC import dataConsumerForm
from Abac_Acp import acpForm
from Abac_AA import AttribAuth
from Abac_NewUser import newUser
from Abac_DO import dataOwnerForm
from Abac_Roles import rolesForm
from Abac_admin import admin
sg.ChangeLookAndFeel("SystemDefaultForReal")
from NewExistingPatient import PaitentForm,newPatient
from NewExistingStaff import StaffForm, newStaff
import pytest
conn = sqlite3.connect('examplemm.db')
c = conn.cursor()
    # return c
# tisha
def welcomeForm():
    welcome = [
        [sg.Text('Welcome to RBAC and ABAC Project', justification='center', font=("", 16), text_color="blue",size=(60, 2))],
        [sg.T("", size=(5, 1)), sg.Button("Admin", key='btnAdmin', size=(15, 1)),sg.T("", size=(5, 1)),
         sg.Button("Hospital Staff", key='btnStaff', size=(15, 1)),sg.T("", size=(5, 1)),
         sg.Button("Patient", key='btnPatient', size=(15, 1))],
        [sg.T("",size=(55, 5)),sg.Button("Close", size=(10, 1), key='btnClose')]
    ]
    welcomeWin = sg.Window("RBAC and ABAC project", layout=welcome, size=(650, 200),resizable=True)
    tblCreate = 0
    # c = conn.cursor()
    while True:
        button, values = welcomeWin.read()
        if button == 'btnAdmin':
            login()
        if button == 'btnPatient':
            PaitentForm()
        if button == 'btnStaff':
            StaffForm()
        if button == 'btnClose':
            welcomeWin.Close()

def login():
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
            rec = c.execute("select username, Pwd, Role from credForm2 where username = ? and Pwd = ? and Role =?" ,(uname, pwd, "Admin"))
            recs = rec.fetchall()
            # rol1 = recs[0][2]
            if recs == '' or recs == []:
                print("No records")
                flag = 1
                break
            else:
                if flag == 0:
                    rolesForm(uname)
        elif button == 'btnReset':
            elems = ['username','password','roleCombo']
            for i in elems:
                loginWin.FindElement(i).Update('')
            loginWin.FindElement('username').set_focus()
        elif button == 'btnExit':
            loginWin.Close()
    loginWin.Close()

    # tisha
welcomeForm()