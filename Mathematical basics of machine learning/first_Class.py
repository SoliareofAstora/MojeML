import numpy as np


def sort(lista):
    while (True):
        sorted = True
        for i in range(len(lista) - 1):
            if lista[i] > lista[i + 1]:
                temp = lista[i]
                lista[i] = lista[i + 1]
                lista[i + 1] = temp
                sorted = False
        if sorted:
            break


x = [1, 6, 4, 2, 7, 9, 4, 2, 7, 4, 32]
x = x * 2
sort(x)

# Stwórz macierz 4x4 typu bool, true na przekątnej.
x = np.empty((4, 4))
np.fill_diagonal(x, 1)
x = (x == 1)

# Wybierz parzyste elementy z losowego wektora 5x5
x = np.random.randint(0, 10, (5, 5))
x = (x % 2 == 0)

# Z wektora [1 .. 10] zastąp parzyste elementy przez 0
x = np.random.randint(0, 100, 100)
x = np.where(x % 2 == 0, 0, x)

# Połącz ze sobą 2 wektory horyzontalniePołącz ze sobą 2 wektory horyzontalnie
x = np.random.randint(0, 100, 10)
x = np.append(x, x)
x = np.array([x, x])

# Utwórz [1 .. 100 ] rozmiaru 10 x 10
x = np.arange(0, 100)
x = x.reshape((10, 10))

# W losowej macierzy 10 x 10 zamień ze sobą 0 i 3 kolumnę (slicing)
x = x[:,[2,1,0,3,4,5,6,7,8,9]]

# Wygeneruj macierz floatów w przedziale 4x4
x = np.random.uniform(0,10,(4,4))

# # Zadania: numpy2 + pkt (za zrobienie większości)
# 1. Odejmij od każdej kolumny średnią kolumny
# 2. Znajdź najbliższą wartość w macierzy (A) do podanej wartości (x)
# 3. W macierzy 20x20 znajdź sumę w bloków, rozmiar bloku (5x5
# Więcej zadań (może częśc odpowiedzi nawet):
#
#  https://www.labri.fr/perso/nrougier/teaching/numpy.100/index.html
#  https://www.machinelearningplus.com/101-numpy-exercises-python/
