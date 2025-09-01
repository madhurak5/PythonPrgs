import datetime
from pyparsing import *
from charm.toolbox.ecgroup import ECGroup
from charm.schemes.pkenc.pkenc_cs98.py import CS98

from charm.toolbox.eccurve import prime192v1
from charm.toolbox.ecgroup import ECGroup

groupObj = ECGroup(prime192v1)
pkenc = CS98(groupObj)

(pk, sk) = pkenc.keygen()

M = b'Hello World!'
ciphertext = pkenc.encrypt(pk, M)

message = pkenc.decrypt(pk, sk, ciphertext)
class User:
    """ User class creates a new user with the given details like name, dob and also prints the user details """
    def __init__(self, first_name, last_name, dob):
        """Initializes the user details"""
        self.first_name = first_name
        self.last_name = last_name
        self.dob = dob
        self.dob_year = int(self.dob[0:4])
        self.dob_month = int(self.dob[4:6])
        self.dob_day = int(self.dob[6:8])


    def prn_usr_details(self):
        """Prints user details"""
        print("First name : ", self.first_name)
        print("Last name : ", self.last_name)
        print("YOu were born on ", self.dob_day, " ", self.dob_month, " ", self.dob_year)


class Staff(User):
    def __init__(self, u, designation):
        super().__init__(u.first_name, u.last_name, u.dob)
        self.designation = designation

    def prn_user_details(self,staff):
        super().prn_usr_details()
        des = staff.designation
        print("U r the ", des)

class Admin:
    def create_user(self):
        pass
    def add_user(self):
        pass
    def delete_user(self):
        pass
    def update_user(self):
        pass
    def revoke_user(self):
        pass
    def assign_permissions(self):
        pass
    def revoke_permissions(self):
        pass
    def update_permissions(self):
        pass
    def get_permissions(self):
        pass

class Superuser:
    pass

class Technician:
    pass

first_name = input("Enter ur first name : ")
last_name = input("Enter ur last name : ")
dob = input("Enter ur date of birth : ")
u = User(first_name, last_name, dob)
des = input("Enter ur designation : ")
s = Staff(u,des)
s.prn_user_details(s)
print(help(User))