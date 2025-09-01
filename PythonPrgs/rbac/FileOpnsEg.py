import fnmatch
import glob
import os
# with open("file1.txt", "w") as f:
#
#     data = "Some content in the file madhura"
#     f.write(data)
# with open("file2.txt", "w") as fr:
#     d = fr.read()
# print(d)
import os
import pathlib
import shutil
import tempfile

listofFolders = os.listdir("C://")
print("With osdir------->")
for i in listofFolders:
    print(i)
print("-"*50)
# for i in listFolders:
#     print(i)
print("With scandir------->")
with os.scandir("C://") as entries:
    for e in entries:
        print(e.name)

print("-"*50)
print("With pathlib------->")
from pathlib import Path
entries = Path("C://")
for en in entries.iterdir():
    print(en.name)
print("-"*50)
# Listing all files in a directory
basepath = "C://0Madhura"
print("Files in ", basepath, "(with os.listdir)----------->")
for entry in os.listdir(basepath):
    if os.path.isfile(os.path.join(basepath, entry)):
        print(entry)

print("-"*50)
print("Files in ", basepath, "(with os.scandir)-----------> ")
with os.scandir(basepath) as entries:
    for entry in entries:
        if entry.is_file():
            print(entry.name)
from pathlib import Path
print("-"*50)

print("Files in ", basepath, "(with pathlib.Path)-----------> ")
basepath = Path(basepath)
files_in_basepath = basepath.iterdir()
for item in files_in_basepath:
    if item.is_file():
        print(item.name)

from pathlib import Path
basepath = Path("C://0Madhura")
print("Files in ", basepath, "(with pathlib.Path using generator expressions)-----------> ")
files_in_basepath2 = (entry for entry in basepath.iterdir() if entry.is_file())
for item in files_in_basepath2:
    print(item.name)
from datetime import datetime
def convert_date(timestamp):
    d = datetime.utcfromtimestamp(timestamp)
    formatted_date = d.strftime("%d %b %Y")
    return formatted_date
def get_files():
    with os.scandir(basepath) as dir_contents:
        for entry in dir_contents:
            info = entry.stat()
            print( f'{entry.name}, \t  Last modified:  {convert_date(info.st_mtime)}')

print("Caling function get_files..............")
get_files()

# os.mkdir("C://0Madhur") #  creates a new dir if it doesn't exist, else raises FileExistsError
# A,ternatve method to create a dir uinsg pathlib
p = Path("C://0Madhur2")
try:
    p.mkdir(exist_ok=True)
except FileExistsError as exc:
    print(exc)
# os.makedirs("2023//Madhura/ML")
# pathlib.Path("2024/Madhura").mkdir(parents=True, exist_ok=True)
#  Filename Pattern matching:
# startswith(), endswith(), fnmatch.fnmatch(), glob.glob(), pathlib.Path.glob()

# Get list of .txt files

pdf,  txt, doc, ppt, xls= 0,0,0, 0, 0
for f_name in os.listdir("C://0Madhura"):
    if f_name.endswith(".pdf"):
        print(f_name)
        pdf += 1
    elif f_name.endswith(".txt"):
        print(f_name)
        txt += 1
    elif f_name.endswith(".docx"):
        print(f_name)
        doc += 1
    elif f_name.endswith(".pptx"):
        print(f_name)
        ppt += 1
    elif f_name.endswith(".xlsx"):
        print(f_name)
        xls += 1
print("No. of .txt files :  ", txt)
print("No. of .pdf files :  ", pdf)
print("No. of .docx files :  ", doc)
print("No. of .pptx files :  ", ppt)
print("No. of .xlsx files :  ", xls)
# using fnmatch is a better option over the string methods that have limited matching abilities.
# fnmatch.fnmatch supports the use of wildcards (*, ?) to match filenames,
for fn in os.listdir("C://0Madhura"):
    if fnmatch.fnmatch(fn, "*.xl*"):
        print(fn)
for fn in os.listdir("C://"):
    if fnmatch.fnmatch(fn, "*Madhura*"):
        print(fn)
import glob
globFiles = glob.glob("*.py")
for n in globFiles:
    print(n)
print("Total no. of files: ", len(globFiles))
c = 0
for f in glob.iglob("**/*.py", recursive=True):
    c +=1
    print(f)
print("Total no. of files found recursively: ", c)
# Difference between glob and iglob: glob returns a list of files, while iglob returns an iterator

# Path.glob()
p = Path("C://0Madhura/")
print("Path.glob() ------->")
for nm in p.glob("*.pd*"):
    print(nm)
print("-"*60)
for dirpath, dirnames, files in os.walk(".", topdown=False):
    print(f"Found directory: {dirpath}")
    for f in files:
        print(f)
print("-"*60)

# from tempfile import TemporaryFile
# with TemporaryFile("w+t") as fp:
#     fp.write("Hello Universe, Thank you so much for the blessings")
#     fp.seek(0)
#     data = fp.read()
# print("-"*60)
#
# with tempfile.TemporaryDirectory() as tmpdir:
#     print("Created temp dir", tmpdir)
#     os.path.exists(tmpdir)
#
# data_file = "C://0Madhur1"
# if os.path.isfile(data_file):
#     os.remove(data_file)
#     print(f"{data_file} is deleted")
# else:
#     print(f'Error: {data_file} is not a valid filename')
#
# try:
#     os.remove(data_file)
#     # print(f"{}")
# except OSError as e:
#     print(f"Error: {data_file} : {e.strerror}")
#
#
# # from pathlib import Path
# # data_file1 = Path("C://0Madhur")
# # try:
# #     data_file1.unlink()
# # except IsADirectoryError as e:
# #     print(f"Error: {data_file1} : {e.strerror}")
#
# # To del directories: os.rmdir(), pathlib.Path.rmdir(), shutil.rmtree()
# trashdir = "C://0Madhur2"
# try:
#     os.rmdir(trashdir)
#     print(f"{trashdir} is deleted")
# except OSError as e:
#     print(f"Error: {trashdir} : {e.strerror}")
import shutil
src = "C://0Madhura/Madhura.jpg"
dest = "C://0Madhur1/"
shutil.copy(src, dest)
print(f"Copied the {src} to {dest}")