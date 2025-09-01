import Email
import os
import PySimpleGUI as sg
import rbac.acl
sg.ChangeLookAndFeel("SystemDefaultForReal")
import sqlite3
import secrets
import string
import time
import datetime
conn = sqlite3.connect('examplemm.db')  # D://PythonPrgs/rbacProj/ D:\PythonPrgs\\rbacProj\
c = conn.cursor()

selReq = ""
def admin(uname):
    vals = []
    CloudServerLayout = [[sg.T("", size=(10, 1))]] + [[sg.Button("Create Cloud Servers", key='crtCldServer', size=(20, 1))]]
    sepLayout0 = [[sg.T("", size=(10, 1))]]
    sepLayout01 = [[sg.T("", size=(10, 1))]]
    sepLayout02 = [[sg.T("", size=(10, 1))]]
    userReqLayout = sepLayout0 + [[sg.Button("Check New User Requests", size=(20, 1), key='btnChkRequest'), sg.Button("Check Role Chg Requests", size=(20,1),key='btnRoleChgReq')]]+ \
                      [[sg.Listbox([], key='lstRequest',size=(20, 4),enable_events=True), sg.Listbox([], key='lstChgRequest',size=(20, 4),enable_events=True)]] +  \
                      [[sg.Button("Approve Requests", size=(20, 1), key='btnAppRequest'), sg.Button("Approve Chg Requests", size=(20, 1), key='btnAppChgRequest')]] + \
                        [[sg.Button("Assign Role", size=(20, 1), key='btnAssignRole')]]
    # , [sg.Button("Check Role Chg Requests", size=(20, 1), key='btnRoleChgReq')]
    DOLayout = sepLayout01 + [[sg.Button("List of Data Owners", key="dataOwners", size=(20, 1))]]
    sepLayout2 = [[sg.T("", size=(10, 1))]]
    DOLayout1 = DOLayout + sepLayout2 + [[sg.Listbox(values=vals, size=(20, 4), key='lstDO')]]
    sepLayout3 = [[sg.T("", size=(10, 1))]]
    DOLayout2 = DOLayout1 + sepLayout3 + [[sg.Button("Delete Selected Data Owner", key='delSelOwn')]] + [[sg.T("", size=(10, 1))]]
    sepLayout4 = [[sg.T("", size=(10, 1))]]
    sepLayout41 = [[sg.T("", size=(10, 1))]]
    DCLayout = [[sg.T("", size=(10, 1))]] + [[sg.Button("List of Data Consumers", key="dataConsumers", size=(20, 1))]]+sepLayout4+\
               [[sg.Listbox(values='', size=(20, 4), key='lstDC')]] + sepLayout41 +  [[
                 sg.Button("Delete Selected Data Consumer", key='delSelCon')]]
    sepLayout5 = [[sg.T("", size=(10, 1))]]
    sepLayout51 = [[sg.T("", size=(10, 1))]]
    AALayout = [[sg.T("", size=(10, 1))]] + [[sg.Button("List of Attribute Authorities", key="attribAuth", size=(20, 1))]] + sepLayout5 +\
                 [[sg.Listbox(values=vals, size=(20, 4), key='lstAA')]] + sepLayout51 + \
                 [[sg.Button("Delete Selected Attribute Authority", key='delSelAA')]]
    sepLayout6 = [[sg.T("", size=(10, 1))]]
    sepLayout61 = [[sg.T("", size=(10, 1))]]
    sepLayout62 = [[sg.T("", size=(10, 1))]]
    FileLayout = sepLayout6 + [[sg.Button("List of Files", key="btnFiles", size=(20, 1))]] + sepLayout61 +\
                   [[sg.Listbox(values='', size=(20, 4), key='lstFil', visible=True)]] + sepLayout62 +\
                   [[sg.Button("Delete Selected File", key='delSelFile')]] + [[sg.Button("Refresh", key='btnRefreshFiles')]]

    adminLay = [
        [sg.TabGroup([[sg.Tab('Cloud Servers', CloudServerLayout), sg.Tab('User requests', userReqLayout),
                       sg.Tab('Data Owners', DOLayout2, title_color='Blue'), sg.Tab('Data Consumers', DCLayout),
                       sg.Tab('Attribute Authorities', AALayout),
                       sg.Tab('Files', FileLayout)]], selected_title_color='green')]
        # [sg.Text("Admin's Account", size=(50, 2), text_color="blue", font=('Times New Roman', 20), justification="center")],
        # [sg.Button("Create Cloud Servers", key='crtCldServer',size=(20, 1))],
        # [sg.Button("Check New User Requests", size=(20, 1), key='btnChkRequest'),sg.Listbox([], key='lstRequest', size=(20, 4), enable_events=True),sg.Button("Approve Requests", size=(20, 1), key='btnAppRequest')],
        # [sg.Button("List of Data Owners",key="dataOwners",size=(20, 1)), sg.Listbox(values= vals,size=(20,4),key='lstDO'),sg.Button("Delete Selected Data Owner", key='delSelOwn')],
        # [sg.Button("List of Data Consumers", key="dataConsumers",size=(20, 1)),sg.Listbox(values='',size=(20,4),key='lstDC'),sg.Button("Delete Selected Data Consumer", key='delSelCon')],
        # [sg.Button("List of Attribute Authorities", key="attribAuth", size=(20, 1)),sg.Listbox(values=vals, size=(20, 4), key='lstAA'), sg.Button("Delete Selected Attribute Authority", key='delSelAA')],
        # [sg.Button("List of Superusers", key="superuser", size=(20, 1)),sg.Listbox(values=vals, size=(20, 4), key='lstSup'),sg.Button("Delete Selected Superuser", key='delSelSup')],
        # [sg.Button("List of Files", key="btnFiles",size=(20, 1)),sg.Listbox(values='',size=(20,4),key='lstFil',visible=True),sg.Button("Delete Selected File",key='delSelFile')],
        # [sg.T(" ", size=(60, 1))],
        # [sg.T(" ", size=(80,1)), sg.Button("Logout", key='logoutAdmin',size=(10, 1))]
    ]

    adminWin = sg.Window("Login to Rbac Project ", layout=adminLay, use_default_focus=True, size=(750, 400), default_element_size=(15, 1))
    credWin_open = True
    delFile = int(0)
    while True:

        button, values = adminWin.read()

        if button is None or button == 'Logout':
            break

        if button == 'crtCldServer':
            filePath = "D://PythonPrgs/Cloud0"
            # exst = os.path.exists(filePath)
            # print("File exists? ", exst)
            # if exst:
            #     fileName = filePath[-1:]
            #     fn = int(fileName) + 1
            #     filePath = filePath[0:-1] + str(fn)
            #     print(filePath)
            #     os.makedirs(filePath,mode=493,exist_ok=False)
            # print("Folder created")
        if button == "btnChkRequest":
            reqs = c.execute("select username from Staff where AARequest = 1 ") #and RoleChange = 0
            recs = reqs.fetchall()
            adminWin.FindElement('lstRequest').Update(recs)
            reqp = c.execute("select username from Patient where AARequest = 1")
            recsp = reqp.fetchall()
            recs.extend(recsp)
            adminWin.FindElement('lstRequest').Update(recs)

        if button == 'lstRequest' and len(values['lstRequest']):
            ids = values['lstRequest']
            selReq = ids[0][0]
            print("Selected ", selReq)
            reqs1 = c.execute("select Email from Staff where username = ?", (selReq,))
            recs1 = reqs1.fetchall()
            reqs11 = c.execute("select Email from Patient where username = ?", (selReq,))
            recs11 = reqs11.fetchall()
            recs1.extend(recs11)
            print(recs1)
            useremail = recs1[0][0]
            print("Useremail : ", useremail)
        if button == 'btnAppRequest':
            print("Selected request to approve : ", selReq)
            reqs1 = c.execute("select username from Staff where username = ?", (selReq,))
            recs1 = reqs1.fetchall()
            print("Records from Staff :", recs1)
            if recs1 == "" or recs1 == []:
                print("No records")
            else:
                for i in recs1:
                    print(i[0])
                    if i[0] == selReq:
                        print("Found match")

                        alphabet = string.ascii_letters + string.digits
                        tok1 = ''.join(secrets.choice(alphabet) for i in range(10))
                        print(tok1)  # YNxLJXhvGF
                        print("Updated Staff")
                        ts = datetime.datetime.now().timestamp()
                        c.execute('update Staff set AARequest = 0, Token = ?, Timestamp = ? where username = ?', (tok1, ts, selReq, ))
                        break
            print("Finding records in Patient table")
            reqs11 = c.execute("select username from Patient where username = ?", (selReq,))
            recs11 = reqs11.fetchall()
            print("Records from Patient :", recs11)
            if recs11 == "" or recs11 == []:
                print("No records in Patient")
            else:
                for i in recs11:
                    if i[0] == selReq:
                        print("Found records in Patient")
                        c.execute("update Patient set AARequest = 0 where username = ?", (selReq,))
                        print("Updated Patient")
                        break
            # recs1.extend(recs11)

            if useremail == "":
                print("User email not specified")
            else:
                # Email.sendmailto(useremail)
                print("Email sent to -> ", useremail)

            print("Updated the request")
            conn.commit()
            reqs = c.execute("select username from Staff where AARequest = 1")
            recs = reqs.fetchall()
            adminWin.FindElement('lstRequest').Update(recs)
            # conn.close()
        if button == 'btnRoleChgReq':
            rc = c.execute("select username from Staff where AARequest = 1 and RoleChange = 1")
            rc1 = rc.fetchall()
            adminWin.FindElement('lstChgRequest').Update(rc1)
        if button == 'btnAppChgRequest':
            idc = values['lstChgRequest']
            selReqc = idc[0][0]

            rc = c.execute("select Role from Staff where AARequest = 1 and RoleChange = 1 and username = ?", (selReqc,))
            rc1 = rc.fetchall()
            print("Selected user asked for role change ", selReqc, rc1[0][0])
            if rc1[0][0] == "Data Owner":
                r = "Data Consumer"
            elif rc1[0][0] == "Data Consumer":
                r = "Data Owner"
            alphabet = string.ascii_letters + string.digits
            tokc = ''.join(secrets.choice(alphabet) for i in range(10))
            print(tokc)  # YNxLJXhvGF
            # print("Updated Staff")
            tsc = datetime.datetime.now().timestamp()
            c.execute('update Staff set AARequest = 0, RoleChange = 0, Token = ?, Timestamp = ?, Role = ? where username = ?',(tokc, tsc,  r,selReqc,))

            # c.execute("update Staff set AARequest = 0, RoleChange = 0, Token = ? where username = ?", (selReqc, ts))
            conn.commit()
        if button == 'dataOwners':
            owners = c.execute("select RoleId from PermissionRole where RoleName = 'Data Owner'")
            recso = owners.fetchall()
            print(recso[0][0])
            lstOwn = c.execute("select UserId from UsersRole where RoleId = ?", (recso[0][0],))
            lstOwners = lstOwn.fetchall()
            print(lstOwners)
            lsto, lsto1 = [], []
            lsto = lstOwners
            print("List of owners ")
            for i in lsto:
                lstStf = c.execute("select username from Staff where userId in (?)",(i[0],))
                lstStaff = lstStf.fetchall()
                print(i[0], lstStaff[0][0])
                lsto1.append(lstStaff[0][0])
            adminWin.Element('lstDO').Update(lsto1)
            if button == 'lstDO' and len(values['lstDO']):
                sg.Popup("Selected ", values['lstDO'])
            # conn.close()
        if button == 'dataConsumers':
            # c = conn.cursor()
            consumers = c.execute("select username from credForm2 where Role = 'Data Consumer'")
            recsc = consumers.fetchall()
            adminWin.Element('lstDC').Update(recsc)
            if button == 'lstDC' and len(values['lstDC']):
                sg.Popup("Selected ", values['lstDC'])
            # conn.close()
        if button == 'attribAuth':
            # c = conn.cursor()
            auth = c.execute("select username from credForm2 where Role = 'Attribute Authority'")
            recaa = auth.fetchall()
            adminWin.Element('lstAA').Update(recaa)
            if button == 'lstAa' and len(values['lstAA']):
                sg.Popup("Selected ", values['lstAA'])
        if button == 'superuser':
            sups = c.execute("select username from credForm2 where Role = 'Superuser'")
            recsup = sups.fetchall()
            adminWin.Element('lstSup').Update(recsup)
            if button == 'lstSup' and len(values['lstSup']):
                sg.Popup("Selected ", values['lstSup'])

        if button == 'delSelOwn':
            dUser = values['lstDO']
            dUname = dUser[0][0]
            print(dUname)
            # c.execute("alter table credForm drop DelUserStatus")
            # c = conn.cursor()
            aftDel = c.execute("select * from credForm2")  # where username = ?",(dUname,))
            aftDelrecs = aftDel.fetchall()
            # print(aftDelrecs)
            delUser = int(1)
            c.execute("delete from credForm2 where username = ?", (dUname,))
            conn.commit()

            aftDel = c.execute("select username from credForm2 where DelUserStatus = 0 and Role = 'Data Owner'")
            aftDelrecs = aftDel.fetchall()
            adminWin.Element('lstDO').Update(aftDelrecs)
            # print(aftDelrecs)
            # conn.close()
        if button == 'delSelCon':
            dUser = values['lstDC']
            dUname = dUser[0][0]
            print(dUname)
            # c.execute("alter table credForm drop DelUserStatus")
            # c = conn.cursor()
            aftDel = c.execute("select * from credForm2")  # where username = ?",(dUname,))
            aftDelrecs = aftDel.fetchall()
            # print(aftDelrecs)
            delUser = int(1)
            c.execute("delete from credForm2 where username = ?", (dUname,))
            conn.commit()
        if button == 'delSelAA':
            dUser = values['lstAA']
            dUname = dUser[0][0]
            print(dUname)
            aftDel = c.execute("select * from credForm2")  # where username = ?",(dUname,))
            aftDelrecs = aftDel.fetchall()
            delUser = int(1)
            c.execute("delete from credForm2 where username = ?", (dUname,))
            conn.commit()
            aftDelc = c.execute("select username from credForm2 where Role = 'Attribute Authority'")  # where username = ?",(dUname,))
            aftDelrecsc = aftDel.fetchall()
            adminWin.Element('lstAA').Update(aftDelrecsc)
        if button == 'delSelSup':
            dUser = values['lstSup']
            dUname = dUser[0][0]
            print(dUname)
            aftDel = c.execute("select * from credForm2")  # where username = ?",(dUname,))
            aftDelrecs = aftDel.fetchall()
            delUser = int(1)
            c.execute("delete from credForm2 where username = ?", (dUname,))
            conn.commit()
            aftDelc = c.execute("select username from credForm2 where Role = 'Superuser'")  # where username = ?",(dUname,))
            aftDelrecsc = aftDel.fetchall()
            adminWin.Element('lstSup').Update(aftDelrecsc)

        if button == 'btnFiles':  # dispFileList()
            colFiles, colFiles1, lstofFiles = [], [], []
            fPath = "D://PythonPrgs/Cloud/"
            lstCldFiles = os.listdir(fPath)
            for f in lstCldFiles:
                fp = fPath + f
                if os.path.isdir(fp):
                    newfPath = fPath + f
                    lstFilesFolder = os.listdir(newfPath)
                    colFiles.append(newfPath)
                colFiles1.append(list(lstFilesFolder))

            for k in colFiles1:
                for m in k:
                    r4 = c.execute("select FileName from FileInfo1 where FileName = ?", (m,))
                    r42 = r4.fetchall()
                    lstofFiles.append(r42)
            adminWin.FindElement('lstFil').Update(lstofFiles)

        if button == 'delSelFile':
            dF = values['lstFil']
            dFile = dF[0]
            print("Selected File to delete : ", dFile[0][0])
            c.execute("update FileInfo1 set DelFileStatus = 1 where FileName = ?", (dFile[0][0],))
            conn.commit()
            aftDel = c.execute('''select FileName from FileInfo1 where DelFileStatus = 0''')
            aftDelrecs = aftDel.fetchall()
            adminWin.FindElement('lstFil').Update(aftDelrecs)

        if button == 'btnRefreshFiles':
            r4 = c.execute("select FileName from FileInfo1 where DelFileStatus = 0")
            r41 = r4.fetchall()
            lstofFiles.append(r41)
            adminWin.FindElement('lstFil').Update(lstofFiles)

        if button == 'btnAssignRole':
            from rbacAdmin import rbacAdminActivity
            rbacAdminActivity(uname)
            # pass
    adminWin.Close()
# admin()
