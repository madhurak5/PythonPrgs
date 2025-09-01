import os
import PySimpleGUI as sg
import sqlite3
import Abac_Acp
import Abac_Acp
from py_abac import PDP, Policy, AccessRequest
import pytest
# @pytest.fixture
# def cur():
#     print("Setting up...")
conn = sqlite3.connect('examplemm.db')
c = conn.cursor()
    # return c,conn

def dataConsumerForm(uname):
    vals =[]
    dcLay = [
        [sg.Text('Login Form for Data Consumers',size=(50, 2),justification="center", font=("", 16),text_color="blue")],
        [sg.Text('Current Data Consumer : '), sg.Text(text=uname, size=(20, 1))],
        [sg.Button('List of Available files', key='btnLstFiles', size=(20, 1)), sg.Listbox(values = vals, key='lstFileNames', size=(40, 5))],
        [sg.Button("Download Selected File ...", key='btnFileSave')],
        [sg.Button("Logout", size=(10, 1), key='btnLogout')]
    ]
    DCWin = sg.Window("Login to Data Consumer Form", layout=dcLay, use_default_focus=False)
    # c = conn.cursor()
    loginWin_open = True
    colFiles =[]
    selFile = ""
    colFiles1 = []
    lstofFiles = []
    policy_json = []
    filenames = ""
    fnc = ""
    fnc0 = ""
    fpc = ""
    fpc1 = ""
    # c = conn.cursor()
    while True:
        button, values = DCWin.read()
        lstofFiles = []
        colFiles = []
        selFile = ""
        colFiles1 = []

        if button == 'btnLstFiles': # dispFileList()
            fPath = "D://PythonPrgs/Cloud/"
            lstCldFiles = os.listdir(fPath)
            for f in lstCldFiles:
                fp = fPath + f
                if os.path.isdir(fp):
                    newfPath = fPath + f
                    lstFilesFolder = os.listdir(newfPath)
                    # print(lstFilesFolder)
                    colFiles.append(newfPath)
                colFiles1.append(list(lstFilesFolder))

            for k in colFiles1:
                for m in k:
                    lstofFiles.append(m)
            # print(len(lstofFiles))
            # print(lstofFiles)

            # owners = c.execute("Select FileName from FileInfo")
            # recs = owners.fetchall()
            DCWin.Element('lstFileNames').Update(lstofFiles)
            # if button == 'lstFileNames' and len(values['lstFileNames']):
            lstofFiles = []
            selFile = values['lstFileNames']

            #     sg.Popup("Selected ", values['lstFileNames'])
        if button == 'btnFileSave':
            fn, r, w, e = "", "","", ""
            acp1 = []
            filenames = values['lstFileNames']
            print("Select file : ",filenames[0])
            fnc = values['lstFileNames']
            fnc0 = fnc[0]

            fpc = c.execute("select FileName, Read, Write, Execute, ACP from FileInfo1 where FileName = ?", (fnc0,))
            fpc1 = fpc.fetchall()
            fn, r, w, e, polr = fpc1[0][0], fpc1[0][1], fpc1[0][2], fpc1[0][3], fpc1[0][4]
            fp = c.execute("select Qualification, Designation, Department, Gender from credForm2 where username = ?",(uname,))
            fpcu = fp.fetchall()
            print(fpcu[0][0],fpcu[0][1],fpcu[0][2],fpcu[0][3])
            from pymongo import MongoClient
            from py_abac import PDP, Policy, Request
            from py_abac.storage import MongoStorage
            import uuid
            from abc import ABCMeta, abstractmethod
            # from typing import TYPE_CHECKING
            # if TYPE_CHECKING:
            #     from py_abac.context import EvaluationContext

            # class AttributeProvider (metaclass=ABCMeta):
            #     # @abstractmethod
            #     def get_attribute_value(self,ace:str,):

            ui = input("Uid : ")
            eff = input("Effect for DC : ")
            import json
            policy_json.insert(0, fpc1[0][4])
            policy_json2 = []
            policy_json = str(policy_json).replace('"','')
            # policy_json2 = [policy_json]
            print("Policy from user interface : ", policy_json)

            policy_json =   [{
                "uid": str(ui),
                "description": "Max and Nina are allowed to create, delete, get any resources only if the client IP matches.",
                "effect": eff,
                "rules": {
                    "subject": [{"$.department": {"condition": "Equals", "value": "Electronics"}},
                                {"$.qualification": {"condition": "Equals", "value": "B.E"}},
                                {"$.designation": {"condition": "Equals", "value": "Professor"}},
                                ],
                    "resource": {"$.filename": {"condition": "RegexMatch", "value":fnc0}},
                    "action": [{"$.method": {"condition": "Equals", "value": "get"}},
                               {"$.method": {"condition": "Equals", "value": "write"}}],
                    "context": {}  # "$.ip": {"condition": "CIDR", "value": "127.0.0.1/35"}
                },
                "targets": {},
                "priority": 0
            }
            ]
            # policy = json.dumps([policy_json],sort_keys=False,indent=4)
            # print("policy after dumping : ", policy)
            policy = policy_json
            # policy = Policy.from_json(policy_json)
            client = MongoClient()
            storage = MongoStorage(client)

            # Add policy to storage
            storage.add(policy)

            print("policy added to storage : ", policy)
            pdp = PDP(storage)
            print(pdp)
            from py_abac.request import AccessRequest
            request_json = {
                "subject": {
                    "id":"",
                    # "attributes": {"department":"Electronics","qualification":"B.E"} #, "gender":"Female", "role":"Data Consumer"}
                    "attributes": {"qualification": fpcu[0][0],"designation":fpcu[0][1], "department": fpcu[0][2]}
                    # {"department":"Computer Science","qualification":"B.E","designation":"Asst. Professor" , "gender":"Female", "role":"Data Consumer"}
                },
                "resource":{
                    "id":"",
                    "attributes": {"filename":fnc0}
                },
                "action": {
                    "id":"",
                    "attributes":{"method":"create"}
                },
                "context":{}
            }

            # # # Parse JSON and create access request object
            # # request = AccessRequest.from_json(request_json)
            req = Request.from_json(request_json)
            # # request = Request.from_json(request_json)
            # # assert
            # # from py_abac.provider.base import AttributeProvider
            # # from py_abac.context import AttributeProvider
            # # ctx = EvaluationContext(req)
            # # policies = storage.get_for_target(ctx.subject_id, ctx.resource_id, ctx.action_id)
            if pdp.is_allowed(req):
                print("Allowed")
                # sg.Popup("Allowed to ", request.action, "the file")
                # print(ctx.get_attribute_value("action", "$.method"))
            else:
                print("Denied")
                # sg.Popup("Denied to", request.action, "the file")
        if button == 'btnLogout':
            DCWin.Close()