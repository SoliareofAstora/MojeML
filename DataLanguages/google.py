import numpy as np
import pandas as pd


html = open("input/gogl.txt")
data = html.read()
data = data.split("\n")

for a in data:
    print(len(a))