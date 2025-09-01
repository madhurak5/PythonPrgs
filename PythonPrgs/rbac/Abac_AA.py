import os
import PySimpleGUI as sg
import sqlite3
conn = sqlite3.connect('examplemm.db') #D://PythonPrgs/rbacProj/ D:\PythonPrgs\\rbacProj\
c = conn.cursor()

def AttribAuth(uname):
    # headings = ['Username','Role']  # the text of the headings
    headings = ['Username', 'Role']
    header = [[sg.Text('  ')] + [sg.Text(h, size=(14, 1)) for h in headings]]
    input_rows = [[sg.Input(size=(15, 1), pad=(0, 0),key='tblReq') for col in range(2)] for row in range(5)]
    exit1 = [sg.Button("Exit", size=(10, 1), key='btnExit')]
    AALay = [
        [sg.Text('Login Form for Attribute Authority', size=(50, 2), justification='center', font=("", 16),text_color='blue')],
        [sg.Text('Current Attribute Authority : '), sg.Text(text=uname, size=(20, 1))],
        [sg.Button("Check Requests", size=(20, 1),key='btnChkRequest'), sg.Listbox([],key='lstRequest', size=(40,4),enable_events=True)],
        [sg.Button("Approve Requests", size=(20, 1), key='btnAppRequest')],
        [sg.Text('  ', size=(50, 1))], [sg.Button("Exit", size=(10, 1), key='btnExit')]
    ]
    AAWin = sg.Window("Login to Attribute Authority Form", layout=AALay , resizable=True, use_default_focus=False)
    loginWin_open = True
    selReq = ''
    appReq = 1
    while True:
        button, values = AAWin.read()
        if button == "btnChkRequest":
            reqs = c.execute("select username from credForm2 where AARequest = 1")
            recs = reqs.fetchall()
            AAWin.FindElement('lstRequest').Update(recs)
        if button == 'lstRequest' and len(values['lstRequest']):
            ids = values['lstRequest']
            selReq = ids[0][0]
            print("Selected ",selReq)
        if button == 'btnAppRequest':
            # appReq = 0
            print("Selected request to approve : ", selReq)
            c.execute('update credForm2 set AARequest = 0 where username = ?', (selReq,))
            sg.Popup(selReq+"'s role has been approved")
            conn.commit()
            reqs = c.execute("select username from credForm2 where AARequest = 1")
            recs = reqs.fetchall()
            # for i in recs:
            #     print(i)
            AAWin.FindElement('lstRequest').Update(recs)
            # conn.close()
        if button == 'btnExit':
            AAWin.Close()
    AAWin.Close()
    return
