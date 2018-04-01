import numpy as np
import datetime
from math import *

Inwokacja = 'Litwo, Ojczyzno moja! ty jesteś jak zdrowie Ile cię trzeba cenić, ten tylko się dowie, Kto cię stracił. Dziś piękność twą w całej ozdobie Widzę i opisuję, bo tęsknię po tobie. Panno święta, co Jasnej bronisz Częstochowy I w Ostrej świecisz Bramie! Ty, co gród zamkowy Nowogródzki ochraniasz z jego wiernym ludem! Jak mnie dziecko do zdrowia powróciłaś cudem (— Gdy od płaczącej matki, pod Twoją opiekę Ofiarowany martwą podniosłem powiekę; I zaraz mogłem pieszo, do Twych świątyń progu Iść za wrócone życie podziękować Bogu —) Tak nas powrócisz cudem na Ojczyzny łono!... Tymczasem, przenoś moją duszę utęsknioną Do tych pagórków leśnych, do tych łąk zielonych, Szeroko nad błękitnym Niemnem rozciągnionych; Do tych pól malowanych zbożem rozmaitem, Wyzłacanych pszenicą, posrebrzanych żytem; Gdzie bursztynowy świerzop, gryka jak śnieg biała, Gdzie panieńskim rumieńcem dzięcielina pała, A wszystko przepasane jakby wstęgą, miedzą Zieloną, na niej zrzadka ciche grusze siedzą.'

text = str.split(Inwokacja)
posortowanie = np.sort(text)

ojczyzny = len(np.where(posortowanie == 'Ojczyzno'))




def primenumbers(x):
    prime = np.ones(x)
    for i in range(1, x):
        for j in range(2, i):
            if i % j == 0:
                prime[i] = 0
    for i in range(x):
        if prime[i] == 1:
            print(i)

primenumbers(20)

dateA = datetime.date(2000, 4, 21)
dateB = datetime.date(2001, 4, 21)

dateB - dateA

a = [1, 2, 3, 3, 4]
b = [3, 4, 6, 0]

e = a + b
np.unique(e)


