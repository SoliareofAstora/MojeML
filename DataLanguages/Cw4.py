import numpy as np


# Napisz definicję funkcji HowManyIntegers(N), która zwróci jak wiele liczb z zakresu [1,N) składa się wyłącznie z cyfr {0,2,7,9}.
#
# Czyli np.
# HowManyIntegers(1) = 0
# HowManyIntegers(10) = 3 # 2,7,9
# HowManyIntegers(28) = 6 # 2,7,9,20,22,27

def howmanyint(N):
    cyfry = [0, 2, 7, 9]
    output = np.array(())
    ilosc = len(cyfry)
    potega10 = 0
    indeksy = np.zeros(int(np.floor(np.log10(N))) + 1, dtype=int)
    output = np.array([])
    liczba = 0

    def updateIndex(i, max):
        indeksy[i] += 1
        if indeksy[i] >= max:
            indeksy[i] = 0
            if i > 0:
                return updateIndex(i - 1, max)
            if i == 0:
                return True
        return False

    while True:
        liczba = 0
        index = 0

        for pot in range(potega10, -1, -1):
            liczba += cyfry[indeksy[index]] * 10 ** pot
            index += 1

        if liczba >= N:
            break

        output = np.append(output, liczba)
        if updateIndex(potega10, ilosc):
            indeksy[0] = 1
            potega10 += 1
    return output


print(howmanyint(1000))



def check_anagram(a,b):
    return sorted(a)==sorted(b)


print(check_anagram("abcd", "dcba") == True)
print(check_anagram("aba", "baa") == True)
print(check_anagram("aba", "ba") == False)
print(check_anagram("tom marvolo riddle ", "i am lord voldemort") == True)

alphabet = 'abcdefghijklmnopqrstuvwxyz '
key = 'zyxwvutsrqponmlkjihgfedcba '


def encode(text, key):
    out = ''
    for i in text:
        out += key[alphabet.index(i)]
    return out


def decore(text, key):
    out = ''
    for i in text:
        out += alphabet[key.index(i)]
    return out


decore(
    encode('ala ma kota', key)
    , key)


##########################################33

def even_numbers_from_list(data):
    return np.array(data)[np.where(np.mod(data, 2) == 0)]
    # return [i for i in data if i % 2 == 0]


print(even_numbers_from_list([1, 2, 3, 4]) == [2, 4])
print(even_numbers_from_list(range(10)) == list(range(0, 10, 2)))
print(even_numbers_from_list(range(1000)) == list(range(0, 1000, 2)))
print(even_numbers_from_list([10, 2, 3, 4, 6, -3, -4]) == [10, 2, 4, 6, -4])


def words_analyze(words):
    return [(i, words[i], len(words[i])) for i in range(len(words))]


print(words_analyze(['tomek', 'jadzia']) == [(0, 'tomek', 5), (1, 'jadzia', 6)])
print(words_analyze([]) == [])


def count_words_starting_with_given_letter(text, letter):
    return {words:1 for words in text.split() if words[0]==letter}


print(count_words_starting_with_given_letter('ola ma kota o imieniu ola', 'o') == {'ola': 2, 'o': 1})
print(count_words_starting_with_given_letter('ola ma kota o imieniu ola', 'k') == {'kota': 1})
print(count_words_starting_with_given_letter('ola ma kota o imieniu ola', 'x') == {})


# Wykorzystując znieżdzenia generatorów list (https://www.python.org/dev/peps/pep-0202/) napisz funkcje, która:
# a) wypisze wszystkie pary (x,y) gdzie 0 < x < n oraz 0 < y < n


def func1(n):
    return [(x,y) for x in range(n) for y in range(n) if x<n if y<n]
    pass

print(func1(3)==[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)])
print(func1(0)==[])

# b) wypisze tylko takie pary dla których x < y

def func2(n):
    return [(x, y) for x in range(n) for y in range(n) if x < y]
    pass

print(func2(3)==[(0, 1), (0, 2), (1, 2)])
print(func2(4)==[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])

# c) wypisze tylko takie pary dla których x > y

def func3(n):
    return [(x, y) for x in range(n) for y in range(n) if x>y]
    pass

print(func3(3)==[(1, 0), (2, 0), (2, 1)])
print(func3(4)==[(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)])