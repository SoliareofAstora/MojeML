import numpy as np


def sum(x):
    out = 0
    for i in range(len(x)):
        out += x[i]
    return out


# sum([1,2,3])

def sub(x, y):
    return x - y


# sub(1,2)

def multi(x):
    out = 1
    for i in range(len(x)):
        out *= x[i]
    return out


# multi([1,2,3,4,5,6])


# str = 'abecadlo'

def reverse(x):
    return x[::-1]


# reverse(str)

def mirror(x):
    return x + x[::-1]


# mirror(str)

def shoot(x):
    out = ''
    for i in range(len(x)):
        out += x[i] + ' '
    return out


# shoot(str)

def IHate3():
    ihate3 = np.arange(30)
    for i in range(1, 30, 3):
        print(ihate3[i:i + 2])


IHate3()

def primenumbers(x):
    prime = np.ones(x)
    for i in range(1, x):
        for j in range(2, i):
            if i % j == 0:
                prime[i] = 0
    for i in range(x):
        if prime[i] == 1:
            print(i)


# primenumbers(20)

def countSentences(x):
    sent = x.split(".")
    for i in sent:
        words = i.split()
        if len(words) > 0:
            print(len(words))


# countSentences("alibaba miała. kotka.")


def miarka(a, b):
    for i in range(a, b + 1, 1):
        print("|....", end="")
    print("|")
    for i in range(a, b + 1, 1):
        print(i, end="")

        for b in range(4 - int(np.log10(i))):
            print(end=" ")


miarka(a = 99,b = 100)

def geometric(n=1, a1=1, q=2):
    out = a1
    for i in range(n):
        out *= q
    return out


geometric(3,3,10)
geometric()

from scipy.stats import hmean


def srednie(x=np.array([])):
    if x.size == 0:
        return 0
    print('srednia arytmetyczna: {}'.format(x.mean()))
    print('srednia ważona: {}'.format(np.average(x, weights=np.ones_like(x))))

    print('srednia harmoniczna: {}'.format(hmean(x)))


# srednie(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
# srednie()


import roman
roman.fromRoman("XI")

def doubleLetters(x):
    out = ""
    bul = True
    for i in range(len(x)):
        if x[i] == " ":
            bul = not bul
            out += " "
        else:
            if i % 2 == bul:
                out += x[i] * 2
            else:
                out += x[i]
    return out

doubleLetters("Ala ma kotełka")

a = input("a")
b = input("b")
c = input("c")
f = np.poly1d([a,b,c])
print(f.roots.real)
