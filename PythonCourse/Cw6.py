import numpy as np
np.__config__.show()
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
tekst = """1 	Adam Kociszewski 	adam.kociszewski@uj.edu.pl
2 	Aleksandra Nowak 	aleksandrairena.nowak@student.uj.edu.pl
3 	Antonino Sota 	antonino.sota@student.uj.edu.pl
4 	Bohdan Samotys 	bohdan.samotys@gmail.com
5 	Jakub Banaśkiewicz 	jakub.banaskiewicz@student.uj.edu.pl
6 	Karolina Bożek 	karolinam.bozek@student.uj.edu.pl
7 	Maksymilian Klimczak 	maks.klimczak@gmail.com
8 	Michał Grzejdziak 	mgrzejdziak@gmail.com
9 	Michał Wróbel 	michal.andrzej.wrobel@student.uj.edu.pl
10 	Piotr Kucharski 	piotr1kucharski@gmail.com
11 	Przemysław Onak 	przemyslaw.onak@student.uj.edu.pl
12 	Wojciech Sabała 	wojciech.sabala96@gmail.com
13 	Piotr Żurek 	Zurcio50@gmail.com """
maile=re.findall(pattern,tekst)
for i in maile:
    print(i)

# Typowym błędem przy szybkim wpisywaniu tekstu jest pisanie drugiej litery wyrazu dużą literą,
# np. SZczecin (zamiast Szczecin) czy POlska (zamiast Polska). Napisz program,
# wykorzystujący funkcję sub i wyrażenia regularne,
# który poprawi wszystkie takie błędy w tekście wprowadzonym przez użytkownika.
# Wyrazy dłuższe niż dwie litery mają być poprawiane automatycznie,
# natomiast o podmianę wyrazu dwuliterowego (np. IT na It)
# program ma pytać użytkownika za każdym razem, gdy na taki natrafi.

tekst = "STudia IT SĄ FAjne"
pattern = r'[A-Z][A-Z][a-z-]+'
maile = re.finditer(pattern,tekst)
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
