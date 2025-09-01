import os
import PySimpleGUI as sg
import sqlite3
conn = sqlite3.connect('examplemm.db') #D://PythonPrgs/rbacProj/ D:\PythonPrgs\\rbacProj\
sg.ChangeLookAndFeel("SystemDefaultForReal")
c = conn.cursor()

def requestRole(uname):
    chk = c.execute("select username from Patient where username = ?", (uname,))
    chkRecs = chk.fetchall()
    if chkRecs == '' or chkRecs == []:
        print("Patient doesn't exist in the system")
    else:
        RoleRequestLayout = [
                    [sg.Text('Role Request Form', size=(50,2), text_color="blue",font=('', 16),justification="center")],
                    [sg.Text('Current Role : ', size=(12, 1)),sg.Combo(('Data Consumer', 'Data Owner'), size=(25, 1),key='cmbCurRole')],
                    [sg.Text('New Role : ', size=(12, 1)),sg.Combo(('Data Consumer', 'Data Owner'), size=(25, 1), key='cmbNewRole')],
                    [sg.T(" ", size=(8,3)), sg.Button("Request Role Change",key='btnReqRoleChg', size=(20, 1)), sg.Button("Reset", key='btnReset', size=(10, 1))]
        ]
        RoleReqWin = sg.Window("RBAC and ABAC Project ", layout=RoleRequestLayout, size=(450, 200),use_default_focus=True,resizable=True)
        while True:
            button, values = RoleReqWin.read()
            # credWin.FindElement('roleCombo').Update(roleToBe)
            elemList = ['cmbCurRole','cmbNewRole']
            if button == 'btnReqRoleChg':
                curRole = values['cmbCurRole']
                newRole = values['cmbNewRole']
                if curRole == newRole:
                    print("The current and new roles cannot be the same. Check ur selection")
                else:
                    recE = c.execute("select username,Role from Patient where username = ? ",(uname,))
                    recExist = recE.fetchall()
                    if recExist == '' or recExist == []:
                        print("No records")
                    else:
                        if curRole != recExist[0][1]:
                            print("Check your current role")
                        else:
                            c.execute("update Patient set Role = ?, AARequest = 1 where username = ?",(newRole, uname, ))
                            print("Updated role")
                            conn.commit()
                            existsUser = 1
            if button == 'btnReset':
                for i in elemList:
                    RoleReqWin.FindElement(i).Update('')
                RoleReqWin.refresh()
                RoleReqWin.FindElement('cmbCurRole').set_focus()
        RoleReqWin.Close()

uname = 'poorvi'
requestRole(uname)