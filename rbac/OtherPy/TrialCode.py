import math
course = "Python for Beginners"
"""
print (len(course))
print (course.upper())
print (course.replace('Beginners', 'Absolute Beginners' ))
print ('python' in course)
print (10 + 3 * 2 ** 2 )
print (round(2.6))
print (abs(-2.8))
print (math.ceil(3.5))

is_hot = True
is_cold = True
if is_hot:
    print ("It's a hot day")
    print ("Drink plenty of water")
elif is_cold:
    print ("It's a cold day")
    print ("Wear warm clothes")
else:
    print ("It's a lovely day Enjoy ur day")
price = 1000000
good_credit = True
if (good_credit):
    down_payment = price * 10/100
    print (price)
else:
    down_payment = price * 20 / 100
print(f"Down payment : ${down_payment}")
high_income = True
crime_rec = False
if(high_income and good_credit):
    print ("1 Eligible for loan")
if (good_credit and not crime_rec):
    print ("Eligible for loan")


temp = int (input ("Enter a temperature : "))
if (temp > 30):
    print ("It's a hot day")
else:
    print ("It's not a hot day")

name = input("Enter a name : ")
chars = int(len(name))
if chars < 3:
    print ("Name must be at least 3 characters")
elif chars > 50:
    print ("Name can be a max of 50 characters")
else:
    print ("Name looks good ")

weight = int(input("Enter ur weight : "))
lbs_or_kgs = input ("Enter (L)bs or (K)gs : ")
if (lbs_or_kgs.upper() == "L") :
    wt = weight * .45
    print (f"You are {wt} lbs")
elif (lbs_or_kgs.upper() == 'K'):
    wt = weight / 0.45
    print(f"You are {wt} kgs")

i = 1
while i <= 5 :
    print ('*' * i)
    i += 1
print ("Done")


sec_no = 9
guess_cnt = 0
guess_limit = 3
while guess_cnt< guess_limit:
    guess = int (input ("Guess : "))
    guess_cnt += 1
    if(guess == sec_no):
        print ("U won!!!")
        break
else:
    print ("Sorry !")


user_input = ""
started = False
stopped = False
while True:
    user_input = input("> ").lower()
    if user_input == "start":
        if started:
            print ("Car is already started")
        else:
            started = True
            print("Car started")

    elif user_input == "stop":
        if not started:
            print("Car is already stopped ")
        else:
            started = False
            print("Car stopped")

        stop = True
    elif user_input == "help":
        print ("start -> to start the car")
        print("stop -> to stop the car")
        print("quit -> to exit application")
    elif user_input == "quit":
        break
    else:
        print("sorry! I don't understand")

for item in 'Python':
    print (item)
for item in [1,2, 3, 4, 5]:
    print (item)
for item in range (10):
    print (item)
for item in range(5,10):
    print(item)
for item in range (1, 10, 2):
    print (item)
price = [10, 20, 30]
cost = 0
for item in price:
    cost += item
print (f"Cost of the items : {cost}")
for x in range (4):
    for y in range (3):
        print (f"({x}, {y})")
nos = [5, 2, 5, 2, 2]
opf = ""
for x in nos:
    for y in range (x):
        opf += "X"
    print (opf)
    opf = ""

print ("End of Fs")
opl = ""
nosl = [2, 2,2, 2, 5]
for x in nosl:
    opl = ""
    for y in range(x):
        opl += "X"
    print (opl)
    opl = ""
print ("")
names = ["John", "Mary", "Nancy", "Mike", "Tom"]
print (names)
print (names[-2])
print (names[2:])

list = [12, 1, 9, 100, 23, 167,33, 63,53, 156]
largest = list[0]
for item in list:
    if item > largest:
        largest = item

print(f"Largest item in the list : {largest}")
matrix = [
            [1,2,3],
            [4,5,6],
            [7,8,9]
]
print (matrix[0][2])
for row in matrix:
    for item in row:
        print (item)
numbers = [5, 2, 1, 100,2,  7, 10, 5, 3, 2, 8,  4, 100]
numbers.append(50)
print(numbers)
numbers.insert(0, 100)
numbers.insert(4, 29)
print(numbers)
numbers.remove(5)
print (numbers)
numbers.pop()
print(numbers)
print("Hello ", numbers.index(1))
print(50 in numbers)
print (numbers.count(100))
numbers.sort()
print(numbers)
numbers.reverse()
print(numbers)
numbers2 = numbers.copy()
print(numbers2)
numbers.append((100))
print(numbers)
uniques = []
for item in numbers:
    if item not in uniques:
        uniques.append(item)
print("Uniques  : ",  uniques)



numbs = (1, 2, 3)  #Tuple
print ("Count : ",numbs.count(3))
print ("First item : ", numbs[0])
# Unpacking
coords = (1, 2,3)

x = coords[0]
y = coords[1]
z = coords[2]  Instead of doing all this, the following code is used to unpack

x, y, z = coords # Unpacks here
print ("x : ", x, "y : ", y, "z : ", z)
customer = {
    "name" : "Madhura",
    "quali" : "Mtech",
    "email" : "madhura@gmail.com",
    "married" : True

}
print (customer.get("nam")) # if key is not there, returns None

customer["name"] = "Madhura M"
print(customer["name"])
customer["dob"] = "May 19"
print(customer["dob"])
phone =  input("Enter ur phone : ")
phone_dict = {
        "1":"One",
        "2" : "Two",
        "3": "Three",
        "4": "Four",
        "5" : "Five",
        "6" : "Six",
        "7" : "Seven",
        "8" : "Eight",
        "9" : "Nine",
        "0" : "Zero"}
op = ""
for ch in phone:
    op += phone_dict.get(ch, "!") + " "
print(op)

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def move(self):
        print ("Move")
    def draw(self):
        print ("Draw")


def emoji(msg):

    words = msg.split(' ')
    print (words)
    emojis = {
             ":)" : "Happy",
            ":(" : "sad"
    }
    op = ""
    for word in words:
        op += emojis.get(word, word) + " "
    return (op)

def greet_usr(name, last):
    print (f"Hi {name} {last}!!!")
    print ("Welcome aboard")


def square(no):
    print (no * no)
    return None

msg = input(">")
e1 = emoji(msg)
print (f"Emoji function : {e1}")
print("Start")
nm = input("Enter ur name : ")
last_nm = input("Enter ur lastname : ")
greet_usr(nm, last = nm)
print("Finish")
print("Square : " ,square(5))

try:
    age = int(input("Age : "))
    income = 20000
    risk = income/age
    print (age)
except ZeroDivisionError:
    print ("Age can't be 0")

except ValueError:
     print ("Invalied Value")



point1 = Point(323, 34)
point1.move()
point1.draw()
point1.x = 10
point1.y = 20
print(point1.x)
point2 = Point(450, 23)
point2.x = 90
print (point2.x)
point3 = Point(100, 200)

print (point3.x)



class Person:
    def __init__(self, name ):
        self.name = name
    def talk(self):
        print (f"It's me, {self.name}")


p = Person("Madhu")
print (p.name)
p.talk()
p = Person("Poorvi")
p.talk()



class Mammal:
    def walk(self):
        print ("Walk")


class Dog(Mammal):
    def bark(self):
        print("Bark")

class Cat(Mammal):
    def meow(self):
        print("Meow Meow")



dog1 = Dog()
dog1.bark()
cat1 = Cat()
cat1.meow()


from convert import *
print (kg_to_lb(90))
print (lb_to_kg(115))

list1 = [4, 350, 100, 2000, 24]
m = find_max(list1)
print (f"Max value : {m}")
"""
# Package Creation
from eComm import shipping
shipping.calc_shipping()
