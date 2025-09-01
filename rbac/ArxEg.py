import os
import zipfile
with zipfile.ZipFile("C://0Madhura/mahesh.zip","r") as zipobj:
    filesinArx = zipobj.namelist()
print(filesinArx)
max = 0
largestFile = ""
for i in filesinArx:
    fileInfo = zipobj.getinfo(i)
    print(i, "--", fileInfo.file_size, ",", fileInfo.compress_size, ",  ", fileInfo.date_time)
    if fileInfo.file_size > max:
        max = fileInfo.file_size
        largestFile = i
print(f"Largest File in the archive is - {largestFile} and has a size of {max}")
files = os.listdir("C://0Madhura/")
print(files)
mm_zip = zipfile.ZipFile("C://0Madhura/mahesh.zip", "r")
# mm_zip.extract("mahesh/Sample_Testing_Resume.pdf", "C://0Madhura/")
# with zipfile.ZipFile("C://0Madhura/Personal/sbi.zip", "r") as pwd_zip:
#     pwd_zip.extractall( path="C://0Madhura/", pwd='Madhura'.encode())

# listSbi = list(os.listdir("C://0Madhura/HealthcareData"))
# print(listSbi)
# # listSbi1 = ['Ppf Apr 21 - Mar 22.pdf', 'Ppf Nov 22 - Apr 23.pdf', 'Savings Apr 21 - Mar 22.pdf', 'Savings Apr 22 - Mar 23.pdf', 'Savings Nov 22 - Apr 23.pdf']
# with zipfile.ZipFile('newSbi.zip', 'w') as new_zip:
#     for fn in listSbi:
#         new_zip.write(fn)
fileList = ['Abac_AA.py', 'Abac_Acp.py']
with zipfile.ZipFile("newZip", "w") as new_zip:
    for fn in fileList:
        new_zip.write(fn)

fileList = ['C://0Madhura/sbi/Ppf Apr 21 - Mar 22.pdf', 'C://0Madhura/sbi/Ppf Nov 22 - Apr 23.pdf', 'C://0Madhura/sbi/Savings Apr 21 - Mar 22.pdf', 'C://0Madhura/sbi/Savings Apr 22 - Mar 23.pdf', 'C://0Madhura/sbi/Savings Nov 22 - Apr 23.pdf']
with zipfile.ZipFile("newSbi", "w") as new_zip:
    for fn in fileList:
        new_zip.write(fn)
# with zipfile.ZipFile("newSbi.zip", "a") as new_zip:
#     new_zip.write("C://0Madhura/Personal/focus-wheel.png")
# import tarfile
# with tarfile.open("C://IET.rar","r") as tar_file:
#     print(tar_file.getnames())

import rarfile
import time
with rarfile.RarFile("C://IET.rar") as rarArc:
    for m in rarArc.infolist():
        print(m.filename, "; Modified Time: ", m.mtime)
import tarfile
file_list = ['C://0Madhura/sbi/Ppf Apr 21 - Mar 22.pdf', 'C://0Madhura/sbi/Ppf Nov 22 - Apr 23.pdf', 'C://0Madhura/sbi/Savings Apr 21 - Mar 22.pdf', 'C://0Madhura/sbi/Savings Apr 22 - Mar 23.pdf', 'C://0Madhura/sbi/Savings Nov 22 - Apr 23.pdf']
print("Creating and reading tar files")
# with tarfile.open("packages.tar", mode="w") as tar:
#     for f in file_list:
#         tar.add(f)
# with tarfile.open("packages.tar", "r") as t:
#     for m in t.getmembers():
#         print(m.name)
# with tarfile.open("packages.tar", "a") as tar:
#     tar.add("C://0Madhura/mahesh.zip")
# with tarfile.open("packages.tar", "r") as t:
#     for m in t.getmembers():
#         print(m.name)
clisby = list(os.listdir("C://1Clisby/"))
# print("Clisby Files --- \n", clisbyFiles)
clisbyFiles = []
for i in clisby:
    file1 = "C://1Clisby/" + i
    clisbyFiles.append(file1)
print("New Clisby Files ", clisbyFiles)
with tarfile.open("clisbyPack.tar.gz", mode="w:gz") as tar:
    for i in clisbyFiles:
        tar.add(i)
print("Added files to clisbyPack.tar")
with tarfile.open("clisbyPack.tar.gz", mode="r:gz") as t:
    for m in t.getmembers():
        print(m.name)
print("Shutil")
import shutil
# shutil.make_archive("data/backup", "tar", "C://0Madhura")

    # process(line)
