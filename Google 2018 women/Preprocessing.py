import pandas as pd

data_dir = 'input/'
df_seeds = pd.read_csv(data_dir + 'WNCAATourneySeeds.csv')
df_tour = pd.read_csv(data_dir + 'WNCAATourneyCompactResults.csv')

# df_seeds.sample(10)
# df_tour.sample(10)


import numpy as np

a = np.array([[1, 2], [3, 4], [5, 6]])

bool_idx = (a > 2)  # Find the elements of a that are bigger than 2;
# this returns a numpy array of Booleans of the same
# shape as a, where each slot of bool_idx tells
# whether that element of a is > 2.

print(bool_idx)  # Prints "[[False False]
#          [ True  True]
#          [ True  True]]"

# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values
# of bool_idx
print(a[bool_idx])  # Prints "[3 4 5 6]"

# We can do all of the above in a single concise statement:
print(a[a > 2])  # Prints "[3 4 5 6]"


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


import numpy as np

x = [1, 6, 4, 2, 7, 9, 4, 2, 7, 4, 32]
x = x * 2

sort(x)
print(x)

x = np.empty((4, 4))
np.fill_diagonal(x, 1)
boolean = (x == 1)
print(boolean)

x = np.random.randint(0, 10, (5, 5))
xp = (x % 2 == 0)
