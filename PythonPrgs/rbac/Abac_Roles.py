import os
import PySimpleGUI as sg
import stat
import sqlite3, pytest
from Abac_admin import admin
from Abac_DC import dataConsumerForm
from Abac_AA import AttribAuth
from Abac_DO import dataOwnerForm

# @pytest.fixture
# def cur():
#     print("Setting up...")
conn = sqlite3.connect('examplemm.db')
c = conn.cursor()
    # return c,conn

aar = ""
r0, r1 = "", ""
def rolesForm(uname):
    rec1 = c.execute("select AARequest, Role from credForm2 where username = ?", (uname,))  # and Role = ?" , selRole
    recs1 = rec1.fetchall()
    r0 = recs1[0][0]
    r1 = recs1[0][1]
    roleLay = [
        [sg.Text('Roles Form', size=(50, 2), justification='center', font=("", 16), text_color='blue')],
        [sg.Text('Username :'), sg.Text(text=uname, size=(20, 1), text_color='blue'), sg.Text("Ur Assigned role ",size=(15, 1)), sg.Text(r1,key="txtRole", size=(20, 1), visible=True, text_color='blue')],
        [sg.T("Choose your role : ",size=(20, 1)),
         sg.InputCombo(('Admin', 'Superuser', 'Technician', 'Data Owner', 'Data Consumer', 'Attribute Authority', 'Doctor'),size=(20, 1),key='roleCombo', enable_events=True),
         sg.Button("Login with role", key="btnLog", size=(20, 1))],
        [sg.T("", size=(41, 1)), sg.Button("Request for another role", key='btnAnRole', size=(20, 1),disabled=False)],
        [sg.T("",size=(41, 1)),sg.Button("Return to Login Form", size=(20, 1), key="btnRet")]
    ]


    roleWin = sg.Window("Role Form", layout=roleLay, use_default_focus=True)

    while True:
        button, values = roleWin.read()
        role, rol,newRole =  "", "", ""
        if button == 'roleCombo':
            if r0 == 1:
                roleWin.FindElement("roleCombo").Update(disabled=True)
                roleWin.FindElement("btnLog").Update(disabled=True)
                roleWin.FindElement("btnAnRole").Update(disabled=True)
            else:
                roleWin.FindElement("roleCombo").Update(recs1[0][1])
                # roleWin.FindElement()
        if button == "btnLog":
            rec = c.execute("select AARequest, Role from credForm2 where username = ?",(uname,))  # and Role = ?" , selRole
            recs = rec.fetchall()
            aar = recs[0][0]
            # rol1 = recs[0][1]
            print("Role from db ", r1)
            print("AA Approval ", aar )
            rol = values['roleCombo']
            print("Role from form ",rol)

            if aar == '0':
                if rol == r1 :
                    if rol == "Admin":
                        admin(uname)
                    elif r1 == "Data Owner":
                        dataOwnerForm(uname)
                    elif r1 == "Data Consumer":
                        dataConsumerForm(uname)
                    elif r1 == 'Attribute Authority':
                        AttribAuth(uname)
                    elif r1 == "Superuser":
                        pass
                    elif r1 == "Technician":
                        pass
                elif (r1 == "" or r1 == None):
                    # c = conn.cursor()
                    # ap = c.execute("select AARequest from credForm2 where username = ?", (uname,))
                    # apprecs = ap.fetchall()
                    # app = apprecs[0][0]
                    # print(app)
                    # if app == 0:
                    print("u haven't been assigned a role yet")
                elif (r1 != "" or r1 != None) and aar == 1:
                    print("Ur request hasn't been approved yet")
            else:
                sg.Popup("Your assigned role has not yet been approved")
                roleWin.Close()

        if button == "btnAnRole":
            newRole = anotherRole(uname, role)
            print("U have been assigned the new role of ", newRole)
            # c = conn.cursor()
            c.execute("update credForm2 set Role = ? where username = ?",(newRole, uname,))
            conn.commit()

        if button == "btnRet":
            roleWin.Close()
    # conn.close()

def anotherRole(uname, role):
    anRoleLay = [
        [sg.Text('Roles Change Form', size=(50, 2), justification='center', font=("", 16), text_color='blue')],
        [sg.Text('Username :'), sg.Text(text=uname, size=(20, 1), text_color='blue')],
        [sg.T("Choose your role : ", size=(20, 1)),
         sg.InputCombo(('Admin', 'Superuser', 'Technician', 'Data Owner', 'Data Consumer', 'Attribute Authority', 'Doctor'),
                       size=(20, 1), key='anRoleCombo'),
         sg.Button("Change Role", key="btnChgRole", size=(20, 1))],
        # [sg.T("", size=(35, 1)), sg.Button("Request for another role", key='btnAnRole', size=(22, 1), disabled=False)],
        [sg.T("", size=(35, 1)), sg.Button("Return ", size=(22, 1), key="btnAnRet")]
    ]

    anRoleWin = sg.Window("Role Form", layout=anRoleLay, use_default_focus=False)
    while True:
        button, values = anRoleWin.read()
        role1 = values['anRoleCombo']
        getuser = c.execute("select Role, AARequest from credForm2 where username = ?", (uname,))
        getuser1 = getuser.fetchall()
        role1 = getuser1[0][0]
        aar = getuser1[0][1]
        if button == "btnChgRole":
            if role1 == "" and aar == 1:
                sg.Popup("U cannot change the role as ur user account has not yet been created")
                anRoleWin.FindElement("anRoleCombo").Update(disabled=True)
            else:
                if values["anRoleCombo"] == role:
                    sg.Popup("U have already been assigned the role, choose another role",button_type='POPUP_BUTTONS_OK')
                    if button == 'POPUP_BUTTONS_OK':
                        anRoleWin.FindElement('anRoleCombo').Update("")
                else:
                    return values['anRoleCombo']

        if button == "btnAnRet":
            anRoleWin.Close()
            return values['anRoleCombo']