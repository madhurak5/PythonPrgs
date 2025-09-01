# import os
# import PySimpleGUI as sg
# import sqlite3
# conn = sqlite3.connect('examplemm.db')
# c = conn.cursor()
# chkElemsTrue = []
# def filePerm(filename):
#     permLay = [
#         [sg.Text('Permissions Form ', size=(50, 2), justification='center', font=("", 16), text_color='blue')],
#         [sg.Text('Filename :'), sg.Text(text=filename, size=(20, 1), text_color='blue')],
#         [sg.Checkbox('Read', key='chkRead'), sg.Checkbox('Write', key='chkWrite'), sg.Checkbox('Execute', key='chkExec')],
#
#         [sg.T("",size=(50, 1)), sg.Button("Apply Permissions", key='btnPerm', size=(22, 1))],
#         [sg.T("",size=(50, 1)),sg.Button("Return to Data Owner Form", size=(22, 1), key="btnRetp")]
#     ]
#
#     permWin = sg.Window("Permissions Form", layout=permLay, use_default_focus=False)
#     # c = conn.cursor()
#     rd, wr, ex = "","",""
#
#     while True:
#         import stat
#         button, values = permWin.read()
#         chkElems = ['chkRead','chkWrite','chkExec']
#
#         if values['chkRead'] == True:
#             rd = stat.S_IROTH
#         if values['chkWrite'] == True:
#             wr = stat.S_IWOTH
#         if values['chkExec'] == True:
#             ex = stat.S_IXOTH
#         # if button == 'btnPerm':
#         #     # for i in chkElems:
#         #     #     chkElemsTrue.append(values[i])
#         #     return (rd, wr, ex)
#         if button == 'btnRetp':
#             permWin.Close()
#             # return (chkElemsTrue)
#             return (rd, wr, ex)
#     return (rd, wr, ex)
#
