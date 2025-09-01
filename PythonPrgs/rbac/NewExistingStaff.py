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
import uuid
# from NewStaff import newStaff
# from NewPatient import newPatient
sg.ChangeLookAndFeel("SystemDefaultForReal")
import pytest
conn = sqlite3.connect('examplemm.db')
c = conn.cursor()

s = 'staff'
def StaffForm():
    newStaffLayout = []
    Staff = [
        [sg.Text('Welcome to RBAC and ABAC Project', justification='center', font=("", 16), text_color="blue",size=(60, 2))],
        [sg.T("", size=(10, 1)), sg.Button("New Staff", key='btnNewStaff', size=(20, 1))],
        [sg.T("", size=(10, 1)), sg.Button("Existing Staff", key='btnExistingStaff', size=(20, 1))],
        [sg.T("",size=(35, 5)),sg.Button("Close", size=(10, 1), key='btnClose')]
    ]
    welcomeWin = sg.Window("Staff Form", layout=Staff, size=(450, 200),resizable=True)
    while True:
        button, values = welcomeWin.read()
        if button == 'btnNewStaff':
            newStaff()
        if button == 'btnExistingStaff':
            login(s)
        if button == 'btnClose':
            welcomeWin.Close()

def newStaff():
    credentialLayout = [
                [sg.Text('User Credentials Form', size=(50,2), text_color="blue",font=('', 16),justification="center")],
                [sg.Text('Username', size=(20, 1)), sg.InputText('',key='user', size=(30,1))],
                [sg.Text('Firstname', size=(20, 1)), sg.InputText('',key='Fname', size=(30,1))],
                [sg.Text('Lastname ', size=(20, 1)), sg.InputText('',key='Lname', size=(30,1))],
                [sg.Text('Password', size=(20, 1)), sg.In(key='Pwd', password_char='*', size=(30, 1))],
                [sg.Text('Confirm Password', size=(20, 1)), sg.In(key='CPwd', password_char='*', size=(30, 1))],
                [sg.Text('Date of Birth', size=(20, 1)), sg.InputText('',key='DoB', size=(30,1))],
                [sg.Text('Gender', size=(20, 1),key='Gender'), sg.Radio('Male', 'gender', default=True, key='genm'), sg.Radio('Female', 'gender', key='genf')],
                [sg.Text("Qualifiation", size=(20,1 )), sg.Combo(('Diploma','Bachelors', 'Masters', 'Doctorate','Specialist'), size=(28, 1),key='Qualification')],
                # [sg.Text("Qualifiation", size=(20, 1)),sg.Combo(('M.B.B.S', 'M.S', 'M.D'), size=(28, 1),key='Qualification')],
                [sg.Text("Designation", size=(20,1 )), sg.Combo(('Patient','Cashier','Asst. Professor', 'Assoc. Professor', 'Professor', 'Professor & HOD', 'Doctor','Researcher','Nurse','Technician','Lab Asst'), size=(28, 1),key='Designation')],
                # [sg.Text("Designation", size=(20, 1)), sg.Combo(('Doctor', 'Assistant', 'Nurse', 'Technician'), size=(28, 1), key='Designation')],
                [sg.Text('No. of Years of Experience', size=(20, 1)), sg.InputText('',key='Experience',size=(30, 1))],
                [sg.Text("")],
                # [sg.Text("Department", size=(20,1)), sg.Combo(('Paediatrics', 'Dental','Gynaecology', 'Radiology', 'Orthopaedic','Cardiology','Dermatology','Neurology'), size=(28, 1), key='Department')],
                [sg.Text("Department", size=(20, 1)), sg.Combo(('Radiology', 'Cardiology', 'Dental', 'BioMedical', 'Orthopaedic','Dermatology','Ophthalmology', 'Neurology', 'Paediatrics', 'Oncology','Hematology', 'Hepatology', 'Hepatology', 'Immunology', 'Gynaecology', 'Obstetrics'),size=(28, 1), key='Department')],
                [sg.Text('Place', size=(20, 1)), sg.InputText('', key='Place', size=(30, 1))],
                [sg.Text('Email', size=(20, 1)),sg.InputText('',key='Email', size=(30,1))],
                [sg.Text("Your role : ", size=(20, 1)),sg.Combo(('Data Owner', 'Data Consumer'),size=(28, 1), key='roleCombo')],
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
        elemList = ['username', 'Fname', 'Lname', 'Pwd', 'CPwd', 'genm', 'genf', 'DoB','Email']
        reqAA = False
        if button == 'createCred':
            # c.execute("create table Staff (username, Fname, Lname, Pwd, CPwd,  DoB, Gender, Qualification, Designation, Experience, Department, Place, Email, Role, AARequest, DelUserStatus)")
            # print("Table created ...")
            if values['genm'] == True:
                gen = 'Male'
            elif values['genf'] == True:
                gen = "Female"
            reqAA = True
            delUser = int(0)
            # chkRecs = c.execute("select * from Staff")
            # chkRecs1 = chkRecs.fetchall()
            # if chkRecs1 == "" or chkRecs == []:
            #     print("Empty Table. Inserting first record")
            #     userid = "1"
            #     c.execute("insert into  Staff (userid, username, Fname, Lname, Pwd, CPwd, DoB, Gender, Qualification, Designation, Experience, Department, Place, Email, Role, AARequest, DelUserStatus) "
            #         "values (?, ?,?,?,?, ?,?,?,?, ?,?,?,?, ?,?,?,?)", (userid, values['user'], values['Fname'], values['Lname'], values['Pwd'], values['CPwd'], values['DoB'],
            #         gen, values['Qualification'], values['Designation'], values['Experience'], values['Department'],
            #         values['Place'], values['Email'], values['roleCombo'], reqAA, delUser,))
            #     conn.commit()
            # else:
            recE = c.execute("select username from Staff where username = ?",(values['user'],))
            recExist = recE.fetchall()
            if recExist == '' or recExist == []:
                print("User can be created")
                userid = str(uuid.uuid4())
                c.execute("insert into  Staff (userid, username, Fname, Lname, Pwd, CPwd, DoB, Gender, Qualification, Designation, Experience, Department, Place, Email, Role, AARequest, DelUserStatus) "
                                  "values (?,?,?,?, ?,?,?,?, ?,?,?,?, ?,?,?,?, ?)",(userid, values['user'], values['Fname'], values['Lname'], values['Pwd'], values['CPwd'], values['DoB'], gen, values['Qualification'], values['Designation'],values['Experience'],values['Department'],values['Place'],values['Email'],values['roleCombo'], reqAA, delUser,))
            #             # # c.execute("create table credForm2 (username, Fname, Lname, Pwd, CPwd, Gender, DoB, Qualification, Designation,Experience, Department, Email, Role, AARequest, DelUserStatus)")
                conn.commit()
                        # print(sid1[0][0])
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