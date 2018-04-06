import numpy as np


# Korzystając z poznanych funkcji (funkcjonałów) proszę napisać własne funkcje max i min
# zwracające odpowiednio największą i najmniejszą wartość listy, np. dla danych [2, 2, 11, 4, 9]
# funkcja powinna zwrócić wartość 11.


class MyList():
    arr = np.array([])

    def __init__(self, data):
        self.arr = np.array(data)

    def max(self):
        if self.arr.size>0:
            temp = self.arr[0]
            for element in self.arr:
                if temp<element:
                    temp = element
        return temp

    def min(self):
        if self.arr.size>0:
            temp = self.arr[0]
            for element in self.arr:
                if temp>element:
                    temp = element
        return temp

data = MyList([2, 2, 11, 4, 9])
data.min()
data.max()


# Proszę napisać funkcję, która będzie obliczać wartości częściowo niepoprawnych wyrażeń:
#
#     28+32+++32++39,
#     4-23--63---1--9,
#     3*4**2**8***1.
#
# Przykładowo dla 28+32+++32++39 funkcja powinna zwrócić wartość 131.



# Napisz program, który będzie sortował dane względem:
#
#     nazwiska,
#     wieku,
#     wzrostu.
#
# Dane przechowujemy jako listę krotek, np.:
# [('John', '20', '90'), ('Jony', '17', '91'), ('Jony', '17', '93'), ('Json', '21', '85'), ('Tom', '19', '80')].
# Możesz użyć tylko raz sort i funkcji lambda.

import pandas as pd

data = pd.DataFrame([('John', '20', '90'),
                     ('Jony', '17', '91'),
                     ('Jony', '17', '93'),
                     ('Json', '21', '85'),
                     ('Tom', '19', '80')]
                    ,columns=['imie','wiek','wzrost'])

data.sort_values(['imie'])
data.sort_values(['wiek'])
data.sort_values(['wzrost'])


# Używając wytwornika list zbuduj listę zawierającą wszystkie liczby podzielne przez
# 4 z zakresu od 1 do n (wartość n wprowadzamy z klawiatury). Następnie wykonaj poszczególne kroki:
#
#     używając funkcji filter usuń z niej wszystkie liczby podzielne przez 8,
#     używając wyrażenia lambda i funkcji map podnieś wszystkie elementy listy
#       (otrzymanej z poprzedniego podpunktu) do sześcianu,
#     używając funkcji reduce i len oblicz średnią arytmetyczną z elementów otrzymanej listy z poprzedniego podpunktu.




# Stwórz trzy listy zawierające po 5 elementów: nazwiska - z nazwiskami pracowników, godziny -
# z liczbą przepracowanych godzin, stawka - ze stawką w złotych za godzinę pracy, np.:
#
# nazwiska = ["Kowalski", "Przybył", "Nowak", "Konior", "Kaczka"],
# godziny = [105, 220, 112, 48, 79],
# stawka = [10.0, 17.0, 9.0, 18.0, 13.0].
#
# Wykorzystując funkcje: zip, map, reduce i filter (oraz, ewentualnie, wytworniki list)
# wyświetl nazwiska i wypłaty (iloczyn stawki godzinowej i liczby przepracowanych godzin)
# tych pracowników, którzy zarobili więcej, niż wyniosła średnia wypłata.


# Napisz własny generator, który będzie zamieniał imiona, pisane małą literą, na imiona pisane z dużej litery, np.:
#
# ['anna', 'ala', 'ela', 'wiola', 'ola'] -> ['Anna', 'Ala', 'Ela', 'Wiola', 'Ola'].
#
# Wypisz wyniki wykorzystując pętlę for i funkcję next. Zmodyfikuj swój generator tak aby
# wybierał tylko imiona n-literowe,
# np.: imiona 3-literowe ['anna', 'ala', 'ela', 'wiola', 'ola'] -> ['Ala', 'Ela', 'Ola']

