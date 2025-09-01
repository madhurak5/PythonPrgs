import stat
import shutil, random
import os
import PySimpleGUI as sg
import sqlite3
sg.ChangeLookAndFeel("SystemDefaultForReal")
from Abac_Acp import acpForm
# from Abac_Perm import filePerm
from pymongo import MongoClient
from py_abac import PDP, Policy, AccessRequest
from py_abac.storage import MongoStorage

import pytest
# @pytest.fixture
# def cur():
#     print("Setting up...")
conn = sqlite3.connect('examplemm.db')
c = conn.cursor()
pol =""
def dataOwnerForm( uname):
    doLay = [
        [sg.Text('Login Form for Data Owners', size=(50, 2), justification='center', font=("", 16),text_color='blue')],
        [sg.Text('Current Data Owner :'), sg.Text(text=uname, size=(20, 1),text_color='blue')],
        [sg.Text('Select a file to Upload', size=(20, 1)), sg.InputText('', key='fileName'),sg.FileBrowse("Browse",key='fileBrws',enable_events=True)],
        [sg.T("", size=(20,1)), sg.Button("Specify Accces Control Policy and Permissions", size=(22, 2), key="btnAcp"), sg.Button("Upload to Cloud", key='upldCld', size=(15, 1),disabled=True)],
        [sg.Button("List the files Uploaded", size=(20, 1),key='btnLstFiles', enable_events=True), sg.Listbox([''],key='lstFiles', size=(40,4)),
         sg.Button('Change Access Control Policy and Permissions', size=(20, 2), key='chgAcpPerm',disabled=True),
            sg.Button('Delete the selected file', size=(20, 1), key='delFile',disabled=True)],
        [sg.T("", size=(70,1)),sg.Button("Logout", size=(10, 1), key='btnLogout')]
    ]

    DOWin = sg.Window("Login to Data Owner Form", layout=doLay, use_default_focus=False)
    lstFilesFolder, lstFilesFolder1, finalList, chk, chk1, multipleVals, attributes , acp   = [], [],[], [],[],[], [], []
    lstFolders,filename ,fn1 ,qual ,des ,dep ,roles,rd, wr,ex, g1, g2, g3, qualm,desigm, deptm, rolem = "","","","", "","","","", "","","","", "", "","","",""
    delFileStat = 0
    while True:
        button, values = DOWin.read()
        print(values)
        if button == 'fileBrws':
            filename = values['fileName']
        if button == 'btnAcp':
            print("Displaying ACP form...")
            filename = str(values['fileName']).rsplit("/", 1)
            fn1 = filename[1]
            print("Assigning ACP to ",fn1)
            acp = acpForm(uname, fn1, "S")
            print("Access control policy formed (got from ACP form) : ", acp)
            DOWin.FindElement('upldCld').Update(disabled=False)
        if button == 'upldCld':
            attributes = [qual,g1, des, g2, dep, g3, roles]
            finalList =[]
            fp1 = "D://PythonPrgs/Cloud/"
            lstFolders = os.listdir(fp1)
            fn = str(values['fileName']).rsplit("/", 1)
            fname = fn[1]
            rndCld = random.randint(0, len(lstFolders)-1)
            cldNo = lstFolders[rndCld]
            from datetime import date,datetime
            curDay = date.today()
            curTime = datetime.now()
            filePath = fp1 + cldNo
            shutil.copy(values['fileName'],str(filePath))
            recE = c.execute("select FileId from ACP where userid = ?", (uname,))
            recExist = recE.fetchall()
            print("fileid : ", recExist)
            Role = ''
            if recExist == '' or recExist == []:
                # fid = c.execute("select FileId from FileInfo1 where FileId = (Select max(FileId) from FileInfo1)")
                #
                # fid1 = fid.fetchall()
                # print("fid1[0][0] : ", fid1[0][0])
                # fileid = fid1[0][0]
                # print(fileid[1:])
                import uuid
                newFileid = str(uuid.uuid4())
                print(newFileid)

                print("uname : ", uname)
                un = c.execute("Select userid from Staff where username = ?", (uname, ))
                un1 = un.fetchall()
                uid = un1[0][0]
                print("Userid of the user uploading the file : ", uid)
                print("File Details that will be inserted into FileInfo1 table : ", newFileid, uid, fname, filePath, cldNo, curTime, delFileStat)
                c.execute("update ACP set FileId = ? where userid = ? and Filename = ?", (newFileid, uid, fname,))
                c.execute('''insert  into FileInfo1 (FileId, userid, FileName, FilePath, Cloud, UploadTime, DelFileStatus) values (?,?,?, ?,?,?, ?)''',(newFileid, uid, fname, filePath, cldNo, curTime, delFileStat, ))
                print("Rec inserted successfully")
                # FileId, userid, newFileid, uid,
                conn.commit()
            for f in lstFolders:
                fp = fp1 + f
                if os.path.isdir(fp):
                    newfPath = fp1 + f
                    lstFilesFolder = os.listdir(newfPath)
                    # print(lstFilesFolder)
                    lstFilesFolder1.append(lstFilesFolder)
            for u in lstFilesFolder1:
                for v in u:
                    finalList.append(v)
        if button == "btnLstFiles":             # dispFileList()
            # owners = c.execute("select FileName from FileInfo1 where Username = ? and DelFileStatus = 0", ("meera",))
            uidb = c.execute("select userid from Staff where username = ?", (uname, ))
            uidb1 = uidb.fetchall()
            uid = uidb1[0][0]
            # print(uid)
            owners = c.execute("select FileId, FileName from FileInfo1 where userid = ? and DelFileStatus = 0", (uid,))
            recs = owners.fetchall()
            print(recs)
            lst =[]
            for row in recs:

                print(row[1])
            for row in recs:
                print(row[0])
            # print("only filenames : ", fnlst)
            # for row in recs:
            DOWin.Element('lstFiles').Update(recs)
            DOWin.FindElement('chgAcpPerm').Update(disabled=False)
            DOWin.FindElement('delFile').Update(disabled=False)
        if button == 'delFile':
            fD = values['lstFiles']
            fDel = fD[0][0]
            fDelN = fD[0][1]
            print("Fileid to be deleted : ", fDel)
            print("File to be deleted : ", fDelN)
            rf = c.execute("select * from FileInfo1 where FileId = ? and FileName = ?", (fDel,fDelN, ))
            rf1 = rf.fetchall()
            print("records for deletion : ", rf1)
            c.execute("Update FileInfo1 set DelFileStatus = 1 where FileId = ? and FileName = ?",(fDel, fDelN,))
            # c.execute("Update FileInfo1 set DelFileStatus = 1 where FileName = ?", (fDel,))
            # print("Deleted File")
            conn.commit()
            owners = c.execute("select FileName from FileInfo1 where userid = ?", (uname,))
            recs = owners.fetchall()
            DOWin.Element('lstFiles').Update(recs)
        if button == 'chgAcpPerm':
            # print("on Hold")
            print("Displaying ACP form...")
            # filename = str(values['fileName']).rsplit("/", 1)
            filename = values['lstFiles']
            fid1 = filename[0][0]
            fn1 = filename[0][1]
            print("filename being assigned permissions now", fn1)
            print("Assigning ACP to ", fn1)
            acp = acpForm(uname, fid1, "C")
            print("Access control policy formed (got from ACP form) : ", acp)
            DOWin.FindElement('upldCld').Update(disabled=False)
        if button == 'btnLogout':
            DOWin.Close()
            break

    DOWin.Close()

# dataOwnerForm()