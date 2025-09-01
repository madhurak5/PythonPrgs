import numpy as np
import scipy as sc
from charm.toolbox.pairinggroup import PairingGroup
import os, sys, stat
import fileinput

filePath = "D://PythonPrgs/csvFiles/admin3.csv"
class Owner:
    def __init__(self):
        print("Owner")
    def getAccessPermission(self):
        return self.perm
    def setAccessPermission(self):
        self.perm = stat.S_IREAD
        os.chmod(filePath, self.perm)
        os.access(filePath,os.W_OK)
o1 = Owner()
o1.setAccessPermission()
per = o1.getAccessPermission()

print(per)