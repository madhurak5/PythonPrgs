from __future__ import print_function
from mysql.connector import errorcode
import sqlite3
import os
import PySimpleGUI as sg
import pymysql.connections
sg.ChangeLookAndFeel("SystemDefaultForReal")
#valid values are ['Black', 'BlueMono', 'BluePurple', 'BrightColors', 'BrownBlue', 'Dark', 'Dark2', 'DarkAmber', 'DarkBlack',
# 'DarkBlack1', 'DarkBlue', 'DarkBlue1', 'DarkBlue10', 'DarkBlue11', 'DarkBlue12', 'DarkBlue13', 'DarkBlue14', 'DarkBlue15',
# 'DarkBlue16', 'DarkBlue17', 'DarkBlue2', 'DarkBlue3', 'DarkBlue4', 'DarkBlue5', 'DarkBlue6', 'DarkBlue7', 'DarkBlue8', 'DarkBlue9',
# 'DarkBrown', 'DarkBrown1', 'DarkBrown2', 'DarkBrown3', 'DarkBrown4', 'DarkBrown5', 'DarkBrown6', 'DarkGreen', 'DarkGreen1', 'DarkGreen2',
# 'DarkGreen3', 'DarkGreen4', 'DarkGreen5', 'DarkGreen6', 'DarkGrey', 'DarkGrey1', 'DarkGrey2', 'DarkGrey3', 'DarkGrey4', 'DarkGrey5',
# 'DarkGrey6', 'DarkGrey7', 'DarkPurple', 'DarkPurple1', 'DarkPurple2', 'DarkPurple3', 'DarkPurple4', 'DarkPurple5', 'DarkPurple6', 'DarkRed',
# 'DarkRed1', 'DarkRed2', 'DarkTanBlue', 'DarkTeal', 'DarkTeal1', 'DarkTeal10', 'DarkTeal11', 'DarkTeal12', 'DarkTeal2', 'DarkTeal3', 'DarkTeal4',
# 'DarkTeal5', 'DarkTeal6', 'DarkTeal7', 'DarkTeal8', 'DarkTeal9', 'Default', 'Default1', 'DefaultNoMoreNagging', 'Green', 'GreenMono', 'GreenTan',
# 'HotDogStand', 'Kayak', 'LightBlue', 'LightBlue1', 'LightBlue2', 'LightBlue3', 'LightBlue4', 'LightBlue5', 'LightBlue6', 'LightBlue7', 'LightBrown',
# 'LightBrown1', 'LightBrown10', 'LightBrown11', 'LightBrown12', 'LightBrown13', 'LightBrown2', 'LightBrown3', 'LightBrown4', 'LightBrown5',
# 'LightBrown6', 'LightBrown7', 'LightBrown8', 'LightBrown9', 'LightGray1', 'LightGreen', 'LightGreen1', 'LightGreen10', 'LightGreen2',
# 'LightGreen3', 'LightGreen4', 'LightGreen5', 'LightGreen6', 'LightGreen7', 'LightGreen8', 'LightGreen9', 'LightGrey', 'LightGrey1',
# 'LightGrey2', 'LightGrey3', 'LightGrey4', 'LightGrey5', 'LightGrey6', 'LightPurple', 'LightTeal', 'LightYellow', 'Material1', 'Material2',
# 'NeutralBlue', 'Purple', 'Reddit', 'Reds', 'SandyBeach', 'SystemDefault', 'SystemDefault1', 'SystemDefaultForReal', 'Tan', 'TanBlue',
# 'TealMono', 'Topanga']
# def chkRole(values):
#     if values[2]=='':
#         print("U din't select any role")
#     else:
#         print("U selected", values[2])

conn = sqlite3.connect('examplemm.db')
c = conn.cursor()

welcomeWindow = False
def chkButton(button):
    if button == "Login":
        print("U have submitted the values")
    elif button == 'Reset':
        print("No submissions")

def dispFileList():
    fPath = "D://PythonPrgs/Cloud/"
    lstCldFiles = os.listdir(fPath)
    for f in lstCldFiles:
        fp = fPath + f
        if os.path.isdir(fp):
            newfPath = fPath + f
            lstFilesFolder = os.listdir(newfPath)
            for i in lstFilesFolder:
                print(i)
def welcomeForm():
    welcome = [
        [sg.Text('Welcome to Rbac Project', justification='center', font=("", 16), text_color="blue",size=(60, 2))],
        [sg.T(" ", size=(10, 1)), sg.Button("New User", key='btnNewUser', size=(20, 1)), sg.Button("Existing User", key='existingUser', size=(20, 1))],
    ]
    # welcomeWin = sg.Window("Login to Rbac project", layout=welcome, size=(600, 300),resizable=True)
    # while True:
    #     button, values = welcomeWin.read()
    #     if button == 'btnNewUser':
    #         newUser()
    #         # pass
    #     if button == 'existingUser':
    #         login()
    #         # pass
    return welcome
    # welcomeWin.Close()

def login():
    loginLay = [
                [sg.Text('Login Form for Existing Users', size=(50, 2), justification='center', font=("", 16),text_color='blue')],
                [sg.Text('Username', size=(20, 1)), sg.InputText('',key='username',focus=True, size=(30, 1))],
                [sg.Text("Password", size=(20, 1)), sg.In(key='password', password_char='*', size=(30, 1))],
                [sg.Text('Role', size=(20, 1), key='Role'),sg.InputCombo(('Admin', 'Superuser', 'Technician', 'Data Owner', 'Data Consumer', 'Attribute Authority'),size=(28, 1),key='roleCombo')],
                [sg.T(" ", size=(20,1))],
                [sg.T(" ", size=(20,1)),sg.Button("Login", key='btnLogin', size=(10,1)), sg.Button("Reset", key='btnReset', size=(10,1))],
                [sg.T(" ", size=(20, 1))],
                [sg.T(" ", size=(50, 1)),sg.Button("Logout", key='btnExit', size=(10,1))]
     ]
    loginWin = sg.Window("Login to Rbac project", layout=loginLay, resizable=True,size=(600, 300))
    loginWin_open = True
    while True:
        button, values = loginWin.read()
        if button is None or button == 'btnExit':
            break
        if button == 'btnLogin':
            uname = values['username']
            pwd = values['password']
            selRole = values['roleCombo']
            print(uname, pwd, selRole)
            rec = c.execute("select username, Pwd, Role from credForm where username = ? and Pwd = ? and Role = ?",(uname, pwd, selRole))
            recs = rec.fetchall()
            if recs == []:
                print("No records")
            else:
                print(recs)
        # conn.close()
            if selRole == 'Admin':
                loginWin.close()
                admin()
            elif selRole == 'Data Owner':
                loginWin.close()
                dataOwnerForm()
            elif selRole == 'Data Consumer':
                loginWin.close
                dataConsumerForm()
        elif button == 'btnReset':
            elems = ['username','password','roleCombo']
            for i in elems:
                loginWin.FindElement(i).Update('')
            loginWin.FindElement('username').set_focus()
        elif button == 'btnExit':
            loginWin.Close()
    loginWin.Close()

def dataOwnerForm():
    doLay = [
        [sg.Text('Login Form for Data Owners', size=(50, 2), justification='center', font=("", 16),text_color='blue')],
        [sg.Text('Select a file to Upload', size=(20, 1)), sg.InputText('', key='fileName'),sg.FileBrowse("Browse",key='fileBrws')],
        [sg.Button("Upload to Cloud", key='upldCld', size=(20, 1))],
        [sg.Button("List the files Uploaded", size=(20, 1),key='btnLstFiles'), sg.Listbox([''],key='lstFiles', size=(40,4))],
        [sg.Button('Delete the selected file', size=(20, 1), key='delFile')],
        [sg.Button("Exit")]
    ]
    DOWin = sg.Window("Login to Data Owner Form", layout=doLay, use_default_focus=False)
    loginWin_open = True
    while True:
        button, values = DOWin.read()
        print(values['fileName'])
        fn = str(values['fileName']).rsplit("/",1)

        if button == 'upldCld':
            # shutil.copy(filePath, folderPath)
            import shutil, random
            fp1 = "D://PythonPrgs/Cloud"
            lstFolders = os.listdir(fp1)
            rndCld = random.randint(0, len(lstFolders)-1)
            print("Cloud No. selected : ",rndCld)
            print("Cloud : ",lstFolders[rndCld])
            from datetime import date,datetime
            curDay = date.today()
            curTime = datetime.now()
            filePath = fp1 + "/" + lstFolders[rndCld]
            shutil.copy(values['fileName'],str(filePath))
            # print(fn, filePath, lstFolders[rndCld], curTime)
            c.execute("insert into FileDataOwner values (?,?,?,?)",(fn, filePath, lstFolders[rndCld], curTime))

        if button == "btnLstFiles":
            # dispFileList()
            fPath = "D://PythonPrgs/Cloud/"
            lstCldFiles = os.listdir(fPath)
            for f in lstCldFiles:
                fp = fPath + f
                if os.path.isdir(fp):
                    newfPath = fPath +f
                    lstFilesFolder = os.listdir(newfPath)
                    for i in lstFilesFolder:
                        print(i)
        if button == 'delFile':
            pass
    DOWin.Close()


def dataConsumerForm():
    vals =[]
    dcLay = [
        [sg.Text('Login Form for Data Consumers',size=(50, 2),justification="center", font=("", 16),text_color="blue")],
        [sg.Button('List of Available files', key='btnLstFiles', size=(20, 1)), sg.Listbox(values = vals, key='lstFileNames', size=(40, 5))],
         [sg.Button("Download Selected File...")],
        [sg.Button("Exit")]
    ]
    DCWin = sg.Window("Login to Data Consumer Form", layout=dcLay, use_default_focus=False)
    loginWin_open = True
    button, values = DCWin.read()

    if button == 'btnLstFiles':
        # dispFileList()
        fPath = "D://PythonPrgs/Cloud/"
        lstCldFiles = os.listdir(fPath)
        for f in lstCldFiles:
            fp = fPath + f
            if os.path.isdir(fp):
                newfPath = fPath + f
                lstFilesFolder = os.listdir(newfPath)
                for i in lstFilesFolder:
                    print(i)
                c = conn.cursor()
                owners = c.execute("Select filenames from FileDataOwner")
                recs = owners.fetchall()
                DCWin.Element('lstFileNames').Update(recs)
                if button == 'lstFileNames' and len(values['lstFileNames']):
                    sg.Popup("Selected ", values['lstFileNames'])

def newUser():
    credentialLayout = [
                [sg.Text('Credentials Form', size=(50,2), text_color="blue",font=('', 16),justification="center")],
                [sg.Text('Username', size=(20, 1)), sg.InputText('',key='username', size=(30,1))],
                [sg.Text('Firstname', size=(20, 1)), sg.InputText('',key='Fname', size=(30,1))],
                [sg.Text('Lastname ', size=(20, 1)), sg.InputText('',key='Lname', size=(30,1))],
                [sg.Text('Password', size=(20, 1)), sg.InputText('', key='Pwd', size=(30,1))],
                [sg.Text('Confirm Password', size=(20, 1)), sg.InputText('', key='CPwd', size=(30,1))],
                [sg.Text('Date of Birth', size=(20, 1)), sg.InputText('',key='DoB', size=(30,1))],
                [sg.Text('Gender', size=(20, 1),key='Gender'), sg.Radio('Male', 'gender', default=True, key='genm'), sg.Radio('Female', 'gender', key='genf')],
                [sg.Text("Qualifiation", size=(20,1 )), sg.Combo(('B.E', 'M.Tech', 'PhD'), size=(28, 1),key='Qualification')],
                [sg.Text("Designation", size=(20,1 )), sg.Combo(('Asst. Professor', 'Assoc. Professor', 'Professor', 'Professor & HOD'), size=(28, 1),key='Designation')],
                [sg.Text('No. of Years of Experience', size=(20, 1)), sg.InputText('',key='Experience',size=(30, 1))],
                [sg.Text("Department", size=(20,1)), sg.Combo(('Computer Science', 'Electronics', 'Mechanical', 'BioMedical'), size=(28, 1), key='Department')],
                [sg.Text('Role', size=(20, 1)), sg.InputCombo(('Admin','Superuser','Technician','Data Owner', 'Data Consumer','Attribute Authority'), size=(28, 2),key='Role')],
                [sg.T(" ", size=(20,1))],
                [sg.T(" ", size=(20,1)), sg.Button("Create",key='createCred', size=(10, 1)), sg.Button("Reset", key='resetCred', size=(10, 1))]
     ]
    credWin = sg.Window("Login to Rbac Project ", layout=credentialLayout, use_default_focus=True,resizable=True)
    credWin_open = True

    while True:
        button, values = credWin.read()
        elemList = ['username', 'Fname', 'Lname', 'Pwd', 'CPwd', 'genm', 'DoB', 'Qualification', 'Designation',
                    'Experience', 'Department', 'Role']
        # un = values['username']
        # print(un)
        # recu = c.execute("select username from credForm where username = ?",(un))
        # recsu = recu.fetchall()
        # canCrt = False
        # if recsu == []:
        #     print("No records")
        #     canCrt = True
        reqAA = False
        if button == 'createCred':
            # if canCrt == True:
            if values['genm'] == True:
                gen = 'Male'
            elif values['genf'] == True:
                gen = "Female"
            reqAA = True
            # c.execute("insert into credForm (?,?)", values (values['username'],reqAA))
            c.execute(
                "insert into  credForm (username, Fname, Lname, Pwd, CPwd, Gender, DoB, Qualification, Designation,"
                "Experience, Department, Role, AARequest) values (?,?,?,?, ?,?,?,?, ?,?,?,?, ?)",
                (values['username'], values['Fname'], values['Lname'], values['Pwd'], values['CPwd'],
                 gen, values['DoB'], values['Qualification'], values['Designation'], values['Experience'],
                 values['Department'], values['Role'], reqAA))
            print("insert successful")
            conn.commit()

        if button == 'resetCred':
            for i in elemList:
                if i == 'genf':
                    values['genf'] = False
                    # values['genm'] = True
                    credWin.FindElement(i).Update('')

                credWin.FindElement(i).Update('')
            credWin.FindElement('username').set_focus()
        # if button == ''
    credWin.Close()


def admin():
    adminLay = [
        [sg.Text("Admin's Account", size=(50, 2), text_color="blue", font=('Times New Roman', 20), justification="center")],
        [sg.Button("Create Cloud Servers", key='crtCldServer',size=(20, 1))],
        [sg.Button("List of Data Owners",key="dataOwners",size=(20, 1)), sg.Listbox(values=['madhura', 'poorvi'],size=(20,4),key='lstDO'),sg.Button("Delete Selected Data Owner", key='delSelOwn')],
        [sg.Button("List of Data Consumers", key="dataConsumers",size=(20, 1)),sg.Listbox(values='',size=(20,4),key='lstDC'),sg.Button("Delete Selected Data Consumer", key='delSelCon')],
        [sg.Button("List of Files", key="files",size=(20, 1)),sg.Listbox(values='',size=(20,4),key='lstFil',visible=True),sg.Button("Delete Selected File",key='delSelFile')],
        [sg.T(" ", size=(60, 1))],
        [sg.T(" ", size=(80,1)), sg.Button("Logout", key='logoutAdmin',size=(10, 1))]
    ]
    adminWin = sg.Window("Login to Rbac Project ", layout=adminLay, use_default_focus=True)
    credWin_open = True
    while True:
        button, values = adminWin.read()
        if button is None or button == 'Logout':
            break
        if button == 'delSelOwn':
            dUser = values['lstDO']
            print(dUser)
            # c.execute("delete from loginForm where username = ?",(dUser))
            selOwner = adminWin.Element('lstDO').Update(values)
            print("DELETED user....")

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
            c = conn.cursor()
            owners = c.execute("Select username from credForm where Role = 'Data Consumer'")
            recs = owners.fetchall()
            adminWin.Element('lstDC').Update(recs)
            if button == 'lstDC' and len(values['lstDC']):
                sg.Popup("Selected ", values['lstDC'])
        if button == 'dataOwners':
            c = conn.cursor()
            owners = c.execute("Select username from credForm where Role = 'Data Owner'")
            recs = owners.fetchall()
            adminWin.Element('lstDO').Update(recs)
            if button == 'lstDO' and len(values['lstDO']):
                sg.Popup("Selected ", values['lstDO'])
        if button == 'files':
            c = conn.cursor()
            owners = c.execute("Select filenames from FileInfo")
            recs = owners.fetchall()
            adminWin.Element('lstFil').Update(recs)
            if button == 'lstFil' and len(values['lstFil']):
                sg.Popup("Selected ", values['lstFil'])
    adminWin.Close()

def deleteUser():
    print("Deleting user...")



def deleteFile():
    print("Deleting file...")


window = sg.FlexForm('Simple data entry form', default_element_size=(40,1))  # begin with a blank form
# window = sg.Window('Simple data entry window').Layout(welcome)
winLayout = welcomeForm()

window = sg.Window("Rbac Project").Layout(winLayout)
button, values = window.read()
print(button)
if button == 'btnNewUser':
    c = conn.cursor()
    # c.execute('''create table credForm (username, Fname, Lname, Pwd, CPwd, Gender, DoB, Qualification, Designation,
    #             Experience, Department, Role, AARequest)''')
    # print("Table created...")
# #     # c.execute('''create table cform ('username', 'Fname', 'Lname', 'Pwd', 'CPwd', 'Gender', 'DoB', 'Qualification', 'Designation',
# #     #                 'Experience', 'Department', 'Role')''')
    newUser()
elif button == 'existingUser':
    login()
# c.execute('''create table loginForm (username, password)''')

