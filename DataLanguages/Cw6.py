import numpy as np

# Napisz funkcję, która będzie zwracała:
#
#     liczbę wierszy,
#     liczbę wyrazów w każdej linii oraz obliczy średnią długość słowa w poszczególnych wierszach,
#     liczbę zdań w pliku.
#
# Funkcja powinna przyjmować jeden argument: nazwę/ścieżkę do pliku.


def split(x):
    return x.split()

def srednia(x):
    return np.array(list(map(len,x))).mean() if len(x)>0 else 0

def zad1(path):
    text = open(path).read()
    wiersze =list(filter(lambda x:len(x)>0, text.split("\n")))
    slowa = list(map(split,wiersze))
    liczbaslow=list(map(len,slowa))
    sredniaslow = list(map(srednia,slowa))
    zdania = list(filter(lambda x:len(x)>0, text.split(".")))

    return {'wiersze': len(wiersze),'liczbawyrazownalinie':liczbaslow,"sredniawyrazow":sredniaslow, 'zdania':len(zdania)}


path = 'input/OgniemMieczem.txt'
ret = zad1(path)


import re
pattern = r'[\w\.-]+@[\w\.-]+'
tekst = "costam@mail.xd a ty wlasnie gozapomniales. innymail@gmail.com Jak szybko go odnalezc?"
maile=re.findall(pattern,tekst)
for i in maile:
    print(i)

def ReverseCopy(path):
    text = open(path).read()
    wiersze = np.array(text.split("\n"))
    rev = np.flipud(wiersze)
    out = open("odwr.txt",'w')
    for i in rev:
        out.write(i+"\n")

ReverseCopy(path)


import pandas as pd
pd.read_csv('input/data.txt').fillna(0).to_csv("filledData.txt",index=False,sep=" ")
