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
# from NewPatient import newPatient
from Abac_Roles import rolesForm
from Login import login
sg.ChangeLookAndFeel("SystemDefaultForReal")
import pytest
conn = sqlite3.connect('examplemm.db')
c = conn.cursor()
import uuid
p = 'patient'
def PaitentForm():
    Patient = [
        [sg.Text('Welcome to RBAC and ABAC Project', justification='center', font=("", 16), text_color="blue",size=(60, 2))],
        [sg.T("", size=(10, 1)), sg.Button("New Patient", key='btnNewPatient', size=(20, 1))],
        [sg.T("", size=(10, 1)), sg.Button("Existing Patient", key='btnExistingPatient', size=(20, 1))],
        [sg.T("",size=(35, 5)),sg.Button("Close", size=(10, 1), key='btnClose')]
    ]
    welcomeWin = sg.Window("Patient Form", layout=Patient, size=(450, 200),resizable=True)
    tblCreate = 0
    # c = conn.cursor()
    while True:
        button, values = welcomeWin.read()
        if button == 'btnNewPatient':
            newPatient()
        if button == 'btnExistingPatient':
            login(p)
        if button == 'btnClose':
            welcomeWin.Close()
#
# def login():
#     loginLay = [
#                 [sg.Text('Login Form for Existing Users', size=(60, 2), justification='center', font=("", 16),text_color='blue')],
#                 [sg.Text('Username', size=(20, 1)), sg.InputText('',key='username',focus=True, size=(30, 1))],
#                 [sg.Text("Password", size=(20, 1)), sg.In(key='password', password_char='*', size=(30, 1))],
#                 [sg.T(" ", size=(20,1))],
#                 [sg.T(" ", size=(20,1)),sg.Button("Login", key='btnLogin', size=(10,1)), sg.Button("Reset", key='btnReset', size=(10,1))],
#                 [sg.T(" ", size=(20, 1))],
#                 [sg.T(" ", size=(50, 1)),sg.Button("Logout", key='btnExit', size=(10,1))]
#      ]
#     loginWin = sg.Window("Login to Rbac project", layout=loginLay, resizable=True,size=(600, 300))
#     flag = 0
#
#     while True:
#         button, values = loginWin.read()
#         if button is None or button == 'btnExit':
#             break
#         if button == 'btnLogin':
#             uname = values['username']
#             pwd = values['password']
#             # selRole = values['roleCombo']
#             rec = c.execute("select username, Pwd, Role from credForm2 where username = ? and Pwd = ?" ,(uname, pwd))
#             recs = rec.fetchall()
#             rol1 = recs[0][2]
#             if recs == '' or recs == []:
#                 print("No records")
#                 flag = 1
#                 break
#             else:
#                 if flag == 0:
#                     rolesForm(uname)
#         elif button == 'btnReset':
#             elems = ['username','password','roleCombo']
#             for i in elems:
#                 loginWin.FindElement(i).Update('')
#             loginWin.FindElement('username').set_focus()
#         elif button == 'btnExit':
#             loginWin.Close()
#     loginWin.Close()

# window = sg.FlexForm('Simple data entry form', default_element_size=(40,1))  # begin with a blank form
# # winLayout = welcomeForm()
# # window = sg.Window("Rbac Project").Layout(winLayout)
# button, values = window.read()
# print("creating table ")

def newPatient():
    credentialLayout = [
                [sg.Text('User Credentials Form', size=(50,2), text_color="blue",font=('', 16),justification="center")],
                [sg.Text('Username', size=(20, 1)), sg.InputText('',key='username1', size=(30,1))],
                [sg.Text('Firstname', size=(20, 1)), sg.InputText('',key='Fname', size=(30,1))],
                [sg.Text('Lastname ', size=(20, 1)), sg.InputText('',key='Lname', size=(30,1))],
                [sg.Text('Password', size=(20, 1)), sg.In(key='Pwd', password_char='*', size=(30, 1))],
                [sg.Text('Confirm Password', size=(20, 1)), sg.In(key='CPwd', password_char='*', size=(30, 1))],
                [sg.Text('Date of Birth', size=(20, 1)), sg.InputText('',key='DoB', size=(30,1))],
                [sg.Text('Gender', size=(20, 1),key='Gender'), sg.Radio('Male', 'gender', default=True, key='genm'), sg.Radio('Female', 'gender', key='genf')],
                # [sg.Text("Qualifiation", size=(20,1 )), sg.Combo(('Diploma','Bachelors', 'Masters', 'Doctorate','Specialist'), size=(28, 1),key='Qualification')],
                # [sg.Text("Qualifiation", size=(20, 1)),sg.Combo(('M.B.B.S', 'M.S', 'M.D'), size=(28, 1),key='Qualification')],
                # [sg.Text("Designation", size=(20,1 )), sg.Combo(('Patient','Cashier','Asst. Professor', 'Assoc. Professor', 'Professor', 'Professor & HOD', 'Doctor','Researcher','Nurse','Technician','Lab Asst'), size=(28, 1),key='Designation')],
                # [sg.Text("Designation", size=(20, 1)), sg.Combo(('Doctor', 'Assistant', 'Nurse', 'Technician'), size=(28, 1), key='Designation')],
                # [sg.Text('No. of Years of Experience', size=(20, 1)), sg.InputText('',key='Experience',size=(30, 1))],
                # [sg.Text("")],
                # [sg.Text("Department", size=(20,1)), sg.Combo(('Paediatrics', 'Dental','Gynaecology', 'Radiology', 'Orthopaedic','Cardiology','Dermatology','Neurology'), size=(28, 1), key='Department')],
                # [sg.Text("Department", size=(20, 1)), sg.Combo(('Radiology', 'Cardiology', 'Dental', 'BioMedical', 'Orthopaedic','Dermatology','Ophthalmology', 'Neurology', 'Paediatrics', 'Oncology','Hematology', 'Hepatology', 'Hepatology', 'Immunology', 'Gynaecology', 'Obstetrics'),size=(28, 1), key='Department')],
                [sg.Text('Place', size=(20, 1)), sg.InputText('', key='Place', size=(30, 1))],
                [sg.Text('Email', size=(20, 1)),sg.InputText('',key='Email', size=(30,1))],
                # [sg.Text("Your role : ", size=(20, 1)),sg.Text("",size=(30, 1), key='roleCombo')], #('Admin', 'Superuser', 'Technician', 'Data Owner', 'Data Consumer', 'Attribute Authority'),
                [sg.Text("Your role : ", size=(20, 1)),sg.Combo(('Data Owner', 'Data Consumer'), size=(28, 1), key='roleCombo')],
                [sg.T(" ", size=(20,1))],
                [sg.T(" ", size=(20,1)), sg.Button("Create",key='createCred', size=(10, 1)), sg.Button("Reset", key='resetCred', size=(10, 1))]
    ]
    credWin = sg.Window("RBAC and ABAC Project ", layout=credentialLayout, use_default_focus=True,resizable=True)
    credWin_open = True
    # c = conn.cursor()
    existsUser = 0
    recExist = ''
    recE = ''
    while True:
        button, values = credWin.read()
        # credWin.FindElement('roleCombo').Update(roleToBe)
        elemList = ['username', 'Fname', 'Lname', 'Pwd', 'CPwd', 'genm', 'genf', 'DoB','Email']
        reqAA = False
        if button == 'createCred':
            # c.execute("create table Patient (username, Fname, Lname, Pwd, CPwd, Gender, DoB, Place, Email, AARequest, DelUserStatus)")
            # print("Table created ...")
            if values['genm'] == True:
                gen = 'Male'
            elif values['genf'] == True:
                gen = "Female"
            reqAA = "1"
            # c = conn.cursor()
            delUser = "0"
            recE = c.execute("select username from Patient where username = ?",(values['username1'],))
            recExist = recE.fetchall()

            if recExist == '' or recExist == []:
                print("User can be created")
                userid1 = str(uuid.uuid1())
                un = values['username1']
                print(userid1, un)
                c.execute('''insert  into Patient(userid, username, Fname, Lname, Pwd, CPwd, Gender, DoB, Place,  Email, Role, AARequest, DelUserStatus) values (?,?,?,?, ?,?,?,?, ?,?,?,?, ? )''', (userid1, un,values['Fname'],values['Lname'], values['Pwd'], values['CPwd'], gen, values['DoB'],values['Place'],values['Email'], values['roleCombo'], reqAA, delUser,))
                conn.commit()

                print("Rec inserted successfully")
                existsUser = 1
            else:
                print("Username already taken")

        if button == 'resetCred':
            for i in elemList:
                if i == 'genm':
                    values['genm'] = True
                elif i == 'genf':
                    values['genf'] = False
                credWin.FindElement(i).Update('')
            credWin.refresh()
            credWin.FindElement('username').set_focus()

    credWin.Close()

# PaitentForm()