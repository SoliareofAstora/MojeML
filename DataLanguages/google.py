import numpy as np
import pandas as pd
import re
import encodings
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from itertools import chain
from collections import Counter


extractNewFile = False
if extractNewFile:
    html = open("input/google/alex.txt")
    input = html.read()
    input = input.split("Wyszukiwano&nbsp;")
    html.close()

    data = pd.DataFrame(columns=['search', 'date'])
    dateNtime = r'>\d{1,2} \S{3} \d{4},.{9}'
    date = r'\d{1,4}'
    day = r'\w{3}'
    a = 50


    # ekst = input[a].split("</div>")[0]
    # czas = re.findall(dateNtime, ekst)

    def monthToNum(shortMonth):

        return {
            'sty': 1,
            'lut': 2,
            'mar': 3,
            'kwi': 4,
            'maj': 5,
            'cze': 6,
            'lip': 7,
            'sie': 8,
            'wrz': 9,
            'paź': 10,
            'lis': 11,
            'gru': 12
        }[shortMonth]

    for a in range(1, len(input), 1):
        tekst = input[a].split("</div>")[0]

        tmp = re.findall(dateNtime, tekst)[0]
        czas = re.findall(date, tmp)

        abcd = tekst.split('<')[1]
        abcd = abcd.split('>')[1]

        data = data.append({'search': abcd
                               , 'date': datetime.datetime(

                int(czas[1]),
                int(monthToNum(re.findall(day, tmp)[0])),
                int(czas[0]),
                int(czas[2]),
                int(czas[3]),
                int(czas[4])
            )}, ignore_index=True)

    data.to_csv('input/google/alex.csv', index=False)

# data( ['search'] ['date']

date_parser = lambda x:datetime.datetime(
    year = int(x[0:4]),
    month = int(x[5:7]),
    day = int(x[8:10]),
    hour = int(x[11:13]),
    minute = int(x[14:16]),
    second = int(x[17:19])
)


data = pd.read_csv('input/google/piter.csv')
data['date'] = data['date'].map(date_parser)


# plotowanie czestosci slow
# data['search'] = [(a.lower()).split() for a in data['search']]
# counter = Counter(chain.from_iterable(data['search']))
#
# common = ['how','to','i','to','the','z','na','do','is'
#     ,'in','and','w','do','a','po','of','-','on','for'
#        ,'o','–','co','czy']
#
# for i in range(100):
#     counter[str(i)] = 0
#
# for i in common:
#     counter[i] = 0
#
# a = counter.most_common(40)
# words = [x[0] for x in a]
# count = [x[1] for x in a]
#
# plt.plot(words,count)
# plt.xticks(rotation = 90)
# plt.show()


data['year'] = data['date'].map(lambda x:x.year)
data['weekday'] = data['date'].map(lambda x:x.weekday())
data['hour'] = data['date'].map(lambda x:x.hour)

# plotowanie gestosci wyszukiwan na dzien i na godzine
# years = data['year'].unique()
# for year in years:
#
#     plt.figure(figsize=(15, 4))
#
#     a = data[data['year']==year].groupby(['weekday','hour'])['search'].count().unstack().fillna(0)
#     sns.heatmap(a,cmap='plasma')
#     plt.title(year)
#     plt.savefig('piter'+str(year))


