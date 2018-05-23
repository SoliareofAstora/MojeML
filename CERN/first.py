import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from trackml.dataset import load_event, load_dataset
from trackml.score import score_event

path_to_train = "input/train_100"

event_name = "event000001000"

hits, cells, particles, truth = load_event(os.path.join(path_to_train, event_name))


def best_stock(data):
    max_price = 0
    answer = ''
    for s in data:
        if data[s] > max_price:
            max_price = data[s]
            answer = s
    return answer


best_stock({'CAC': 10.0,'ATX': 390.2,'WIG': 1.2})

range(9)

data = {'CAC': 10.0,'ATX': 390.2,'WIG': 1.2}
for i in data:
    print(i)
    print(data[i])

data = [1,3,2,5,6,7,10,100]
for i in data:
    print(i)

for i in range(len(data)):
    print(data[i])