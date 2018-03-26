# Napisz definicję funkcji HowManyIntegers(N), która zwróci jak wiele liczb z zakresu [1,N) składa się wyłącznie z cyfr {0,2,7,9}.
#
# Czyli np.
# HowManyIntegers(1) = 0
# HowManyIntegers(10) = 3 # 2,7,9
# HowManyIntegers(28) = 6 # 2,7,9,20,22,27




def HowManyIntegers(N):
    cyfry = {0, 2, 7, 9}
    output = np.array(())
    for potega in range(floor(log10(N)) + 1):
        for i in cyfry:
            liczba = i * 10 ** potega
            if liczba < N:
                output = np.append(output, liczba)
            for pozostale in range(potega):
                for j in cyfry:
                    liczba +=j*10**pozostale
                    if liczba < N:
                        output = np.append(output, liczba)
                    liczba-=j*10**pozostale

            if liczba < N:
                output = np.append(output, liczba)
            else:
                return output


HowManyIntegers(28)