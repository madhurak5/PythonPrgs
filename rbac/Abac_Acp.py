import os
import PySimpleGUI as sg
import stat
import json
import sqlite3
sg.ChangeLookAndFeel("SystemDefaultForReal")
from py_abac.policy import Policy
from pymongo import MongoClient
from py_abac.storage import MongoStorage
from rbacAbacPolicy import rbacAbacPolicyStore
from random import randint
policy = ""
rbacAbacPolicyStore = ""
import pytest
# @pytest.fixture
# def cur():
#     print("Setting up...")
conn = sqlite3.connect('examplemm.db')
c = conn.cursor()
kws = {'AND', 'OR'}
qua = ["MBBS", "MS", "MD"]

dep = ["Radiology", "Cardiology", "Oncology"]
attribs = {'Designation', 'Qualification', 'Department', 'Role'}
fname = ""

ge, c1,u1, d1 = "","","",""
q, d = "",""

def acpForm(user, filename, spec):
    userid = user
    global acp
    pCreateTime = ""
    pstatus = ""
    acp = ""
    userid, fid = "",""
    acpLay = [
        [sg.Text('Access Control Policy Form ', size=(50, 2), justification='center', font=("", 16), text_color='blue')],
        [sg.Text('Filename :'), sg.Text(text=filename, size=(20, 1), text_color='blue')],
        # [sg.Checkbox('Department', key='chkDept'), sg.Checkbox('Role', key='chkRole'), sg.Checkbox('Qualification', key='chkQual'), sg.Checkbox('Designation',key='chkDesig')],
        [sg.Text("Qualifiation", size=(20, 1)),sg.Checkbox('Values Set', key='qualVals',enable_events=True),
         sg.Combo(('M.B.B.S', 'M.S', 'M.D'), size=(28, 1), key='Qual')],
        # [sg.T("",size=(20, 1)), sg.Checkbox("And", key='chkAndQ', size=(10,1)),sg.Checkbox("Or", key='chkOrQ', size=(10,1))],
        [sg.Text('Logical Gate', size=(20, 1), key='logGate1'), sg.Radio('And', 'Grp1', default=False, key='and1'),sg.Radio('Or', 'Grp1', key='or1')],
        [sg.Text("Designation", size=(20, 1)),sg.Checkbox('Values Set', key='desigVals',enable_events=True),
         sg.Combo(('Doctor', 'Assistant', 'Nurse', 'Technician'), size=(28, 1),key='Design')],
        [sg.Text('Logical Gate', size=(20, 1), key='logGate2'), sg.Radio('And', 'Grp2', default=False, key='and2'),sg.Radio('Or', 'Grp2', key='or2')],
        [sg.Text("Department", size=(20, 1)), sg.Checkbox('Values Set', key='deptVals',enable_events=True),
         sg.Combo(('Radiology', 'Cardiology', 'Dental', 'BioMedical', 'Orthopaedic','Dermatology','Ophthalmology', 'Neurology', 'Paediatrics', 'Oncology'),
                  size=(28, 1), key='Dept')],
        [sg.Text('Logical Gate', size=(20, 1), key='logGate3'), sg.Radio('And', 'Grp3', default=False, key='and3'),sg.Radio('Or', 'Grp3', key='or3')],
        [sg.Text('Role', size=(20, 1)),sg.Checkbox('Values Set', key='roleVals',enable_events=True),
         sg.Combo(('Admin', 'Patient','Superuser', 'Doctor','Technician', 'Data Owner', 'Data Consumer', 'Attribute Authority', 'Nurse','Cashier'),
                  size=(28, 2), key='Role')],
        # sorted(iterable=True)
        [sg.Text("Assign File Permissions : ", size=(20, 1)),sg.Checkbox('Get', key='chkGet'),
         sg.Checkbox('Create', key='chkCreate'),sg.Checkbox('Update', key='chkUpdate'),sg.Checkbox('Delete', key='chkDelete')],
        [sg.Text('Permissions', size=(20, 1), key='perm'), sg.Radio('Allow', 'Grp4', default=False, key='allow'),sg.Radio('Deny', 'Grp4', key='deny')],
        [sg.T("", size=(50, 1)), sg.Button("Specify Access Control Policy", key='btnAcp', size=(22, 1))],
        [sg.T("",size=(50, 1)),sg.Button("Return to Data Owner Form", size=(22, 1), key="btnRet")]
    ]

    acpWin = sg.Window("Access Control Policy Form", layout=acpLay, use_default_focus=False)
    # c = conn.cursor()
    g1, g2, g3 = "", "", ""
    fid = ""
    qualm, desigm, deptm, rolem = "", "", "", ""
    rd, wr, ex = "", "", ""
    eff = ""
    permGet, permCreate, permDel, permUpdate = "","", "", ""
    chkElemsTrue, chkGatesTrue, chkPerms, chkPermSel, multipleVals = [], [], [], [], []
    while True:
        button, values = acpWin.read()
        chkPerms = ['chkGet', 'chkCreate', 'chkUpdate', 'chkDelete']
        q =  acpWin.FindElement("qualVals").get()
        if q:
            Q = "anyof"
            acpWin.FindElement("Qual").Update(disabled=True)
        else:
            Q = acpWin.FindElement("Qual").get()

        if values['desigVals'] == True:
            Dg = "anyof"
            acpWin.FindElement("Design").Update(disabled=True)
        else:
            Dg = values['Design']
        if values['desigVals'] == False:
            acpWin.FindElement("Design").Update(disabled=False)

        if values['deptVals'] == True:
            Dp = "anyof"
            acpWin.FindElement("Dept").Update(disabled=True)
        else:
            Dp = values['Dept']
        if values['deptVals'] == False:
            acpWin.FindElement("Dept").Update(disabled=False)

        if values['roleVals'] == True:
            R = "anyof"
            acpWin.FindElement("Role").Update(disabled=True)
        else:
            R = values['Role']
        if values['roleVals'] == False:
            acpWin.FindElement("Role").Update(disabled=False)

        G = acpWin.FindElement('chkGet').get()
        C = acpWin.FindElement('chkCreate').get()
        U = acpWin.FindElement('chkUpdate').get()
        D = acpWin.FindElement('chkDelete').get()
        Allow = acpWin.FindElement("allow").get()
        Deny = acpWin.FindElement("deny").get()
        action =""
        if Allow == True:
            action = "allow"
        elif Deny == True:
            action = "deny"
        import datetime
        ts = datetime.datetime.now().timestamp()
        uploadTime = datetime.datetime.fromtimestamp(ts).isoformat()
        status = 0
        ga1 = acpWin.FindElement("and1").get()
        go1 = acpWin.FindElement("or1").get()
        gate1, gate2, gate3 = "","",""
        if ga1 == 1:
            gate1 = " AND "
        elif go1 == 1:
            gate1 = " OR "
        ga2 = acpWin.FindElement("and2").get()
        go2 = acpWin.FindElement("or2").get()
        if ga2 == 1:
            gate2 = " AND "
        elif go2 == 1:
            gate2 = " OR "
        ga3 = acpWin.FindElement("and3").get()
        go3 = acpWin.FindElement("or3").get()
        if ga3 == 1:
            gate3 = " AND "
        elif go3 == 1:
            gate3 = " OR "
        ge, c1, u1, d1 = "", "", "", ""
        if G:
            ge = "get"
        if C:
            c1 = "create"
        if U:
            u1 = "update"
        if D:
            d1 = "delete"


        # acp = ""
        Qa, Dga, Dpa, Ra = "","","",""
        if button == 'btnAcp':
            if Q:
                Qa = "Qualification=" + Q

            if Dg:
                Dga = "Designation=" + Dg
            if Dp:
                Dpa = "Department=" + Dp
            if R:
                Ra = "Role=" + R
            acp = "(" +Qa + gate1 + Dga  + ") "+ gate2 + Dpa + gate3 + Ra

            uidb = c.execute("select userid from Staff where username = ?", (user, ))
            uidb1 = uidb.fetchall()
            if uidb1 == "" or uidb1 == []:
                print("No records ")
            else:
                userid = uidb1[0][0]

            fidb = c.execute("select FileId, FileName from FileInfo1 where userid = ?",(uidb1[0][0],))
            fidb1 = fidb.fetchall()

            if fidb1 == "" or fidb1 == []:
                print("No records")
            else:
                print(fidb1)
                fid = fidb1[0][0]
            print("File id from FileInfo1 Table : .... ", fid)

            import uuid
            polid = uuid.uuid4()
            print("Policy Id : ", polid)
            from datetime import date, datetime
            curDay = date.today()
            pCreateTime = datetime.now()
            pstatus = 0

            # c.execute("insert into  ACP (userid, FileId, FileName, Qualification, Department, Designation, Role, FGet, FCreate, FUpdate, FDelete, Action,  Policy, PCreateTime,  PStatus) values (?,?,?, ?, ?,?,?,?, ?,?,?, ?,?,?,?)", (userid, fid, filename, Q, Dp, Dg, R, G, C, U, D, action, acp, pCreateTime, pstatus,))
            # conn.commit()
            import re
            res = re.search('\(([^)]*)', acp).group(1)
            print("Single search : ", res)
            res1 = re.findall('\(([^)]*)', acp)
            print("Trying with * :",res1)
            k = ""
            l = ""
            for i in res1:
                j = i.split(" ")
                for k in j:
                    if k in attribs:
                        print("found attrib : ", k)
                    if k not in kws:
                        if k.find("=") != -1:
                            l = k.split("=")
                            for m in range (0, len(l)):
                                if l[m] in attribs:
                                    print("attrib found", l[m])
                                else:
                                    print("attrib value", l[m])
                        else:
                            print(k)
                    flag = 0
                    if Q == "anyof" or D == "anyof":
                        if q in qua or d in dep:
                            print("allowed")
                            flag = 0
                        else:
                            print("NOt allowed")
                            flag = 1

                # c.execute("insert into  ACP (userid, FileId, FileName, Qualification, Department, Designation, Role, FGet, FCreate, FUpdate, FDelete, Action,  Policy, PCreateTime,  PStatus) values (?,?,?, ?, ?,?,?,?, ?,?,?, ?,?,?,?)",
                #     (userid, fid, filename, Q, Dp, Dg, R, G, C, U, D, action, acp, pCreateTime, pstatus,))
                # conn.commit()
        if spec == "C":
            print("ACP change request")
            print(acp)
            r1 = c.execute("select FileName from FileInfo1 where FileId = ?",(filename,))
            r11 = r1.fetchall()
            fn2 = r11[0][0]
            print("Filename name retrieval :" , fn2)

            print(userid, filename, fn2, Q, Dp, Dg, R, G, C, U, D, action, acp, pCreateTime, pstatus)
            c.execute("insert into  ACP (userid, FileId, FileName, Qualification, Department, Designation, Role, FGet, FCreate, FUpdate, FDelete, Action,  Policy, PCreateTime,  PStatus) values (?,?,?, ?, ?,?,?,?, ?,?,?, ?,?,?,?)", (userid, filename, fn2, Q, Dp, Dg, R, G, C, U, D, action, acp, pCreateTime, pstatus,))
            conn.commit()
        elif spec == "S":
            print("ACP specify request")

            print(acp)
            print(userid, fid, filename, Q, Dp, Dg, R, G, C, U, D, action, acp, pCreateTime, pstatus)
            c.execute("insert into  ACP (userid, FileId, FileName, Qualification, Department, Designation, Role, FGet, FCreate, FUpdate, FDelete, Action,  Policy, PCreateTime,  PStatus) values (?,?,?, ?, ?,?,?,?, ?,?,?, ?,?,?,?)", (userid, fid, filename, Q, Dp, Dg, R, G, C, U, D, action, acp, pCreateTime, pstatus,))
            conn.commit()
        if button == 'btnRet':

            print("Printing ACP before returning to DO form : ", acp)
            acpWin.Close()
            return acp


# acpForm("mm","D:/NPTEL/Assignments Calcs.xlsx")