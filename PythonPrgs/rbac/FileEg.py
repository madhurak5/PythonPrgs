# import fileinput
# import sys
# files = fileinput.input()
#
# for line in files:
#     if fileinput.isfirstline():
#         print(f'\n ---- Reading {fileinput.filename()} -----')
#     print('->' + line, end='')
# print()
# filename1= input("Enter filename")
# f1 = open(filename1, "w+")

# f = open('a.dat', 'w+')
# # f.close()
# # f.read()
# # f.write("Welcome to the first file")
# # print(list(f))
# f.write('Welcome to first line of my file.')
# f.write('Here comes the second line of my file. ')
# f.write('Third line. ')
# f.seek(0)
# t=f.readlines()
# print(t)
print(any(x > 0 for x in [1,2,-3,-4]))
print(any(x > 0 for x in [-1,-2,-3,-4]))
print(all(x > 0 for x in [1,2,-3,-4]))
x = 10
print(eval('x+1'))
print(globals())
print(locals())
print(max([2,3,56,0,-9]))
print(hex(147))
print(bin(89))
# y = x+10
# print(y)
# x = '14'
# y = x+ 10
# print(y)
# print((5/(6-(5+1))))  # ZeroDivisionError: division by zero
print((5/(6-5+1)))  # 2.5
import math
print(math.pi)
import random
import statistics
print(random.random())
data = [12.5, 71.75, 13.25, 10.75]
print(statistics.mean(data))

print(math.log(1000,2))
a = 10
b = 20
a, b = b, a
print(a, b)

# def table(n):
#     i = 1
#     while i <= 10:
#         print(n, '*', i, '=', n*i)
# table(3)

print('C:\What is your \name')
print(r'C:\What is your \name')

# class MyError(Exception):
#     def __init__(self, value):
#         self.value = value
#     def __str__(self):
#         return repr(self.value)
#
#
# m = MyError(e)
# print(m)
print(chr(68))
import sys
print(sys.version_info)
import os
print(os.getcwd())
print(os.getpid())
x = 20
assert x >=20, "x is not greater than or equal to 20"
# x = 10
# del x
# print(x)

# try:
#     a = int(input("Enter a number"))
#     b = a - 5
#     c = a//b
#     raise ZeroDivisionError("cannot divide")
#     # raise ValueError("Wrong input")
# except ZeroDivisionError:
#     raise
# except ValueError:
#     raise
# finally:
#     print(c)

a = lambda a1: a1 * 4
for i in range(1,3):
    print(a(i))
print(47 or 56)

def gen():
    for i in range(4):
        yield  i * i
g = gen()
for a in g:
    print(a)
print(~(25))
print(4|1)
print('h' in 'hello')
import pandas as pd
