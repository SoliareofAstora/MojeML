
def DiceIndex(confMatrix):
    p = confMatrix[1, 1] / confMatrix[:, 1].sum()
    r = confMatrix[1, 1] / confMatrix[1, :].sum()
    return 2*(p*r)/(p+r)
