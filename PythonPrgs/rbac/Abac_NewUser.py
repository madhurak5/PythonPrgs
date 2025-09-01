# from DecisionTreeEg import assignRole
import os
import PySimpleGUI as sg
sg.ChangeLookAndFeel("SystemDefaultForReal")
import sqlite3
conn = sqlite3.connect('examplemm.db') #D://PythonPrgs/rbacProj/ D:\PythonPrgs\\rbacProj\
c = conn.cursor()
# roleToBe = assignRole()
# print(roleToBe)
def newUser():
    credentialLayout = [
                [sg.Text('User Credentials Form', size=(50,2), text_color="blue",font=('', 16),justification="center")],
                [sg.Text('Username', size=(20, 1)), sg.InputText('',key='username', size=(30,1))],
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
            reqAA = True
            # c = conn.cursor()
            delUser = int(0)
            recE = c.execute("select username from Patient where username = ?",(values['username'],))
            recExist = recE.fetchall()
            print(recExist)
            if recExist == '' or recExist == []:
                print("User can be created")

                c.execute("insert into  Patient (userid, username, Fname, Lname, Pwd, CPwd, Gender, DoB, Place, Email, AARequest, DelUserStatus) "
                          "values (?,?,?,?,?, ?,?,?,?, ?,?,?)",
                          (values['userid'],values['username'], values['Fname'], values['Lname'], values['Pwd'],
                           values['CPwd'],gen, values['DoB'], values['Place'],values['Email'],reqAA, delUser,))
                # c.execute("create table credForm2 (username, Fname, Lname, Pwd, CPwd, Gender, DoB, Qualification, Designation,Experience, Department, Email, Role, AARequest, DelUserStatus)")
                conn.commit()
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

    # conn.close()
    credWin.Close()
# newUser()