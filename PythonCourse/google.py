import pandas as pd
import re
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
from itertools import chain
from collections import Counter


extractNewFile = False
if extractNewFile:

    html = open("input/google/janek.txt")
    input = html.read()
    input = input.split("Wyszukiwano&nbsp;")
    html.close()

    data = pd.DataFrame(columns=['search', 'date'])
    dateNtime = r'>\d{1,2} \S{3} \d{4},.{9}'
    date = r'\d{1,4}'
    day = r'\w{3}'

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

    data.to_csv('input/google/janek.csv', index=False)

# data( ['search'] ['date'] )

date_parser = lambda x:datetime.datetime(
    year = int(x[0:4]),
    month = int(x[5:7]),
    day = int(x[8:10]),
    hour = int(x[11:13]),
    minute = int(x[14:16]),
    second = int(x[17:19])
)


data = pd.read_csv('input/google/alex.csv')
data['date'] = data['date'].map(date_parser)


counter = Counter(chain.from_iterable(data['search'].map(lambda x: (x.lower()).split())))
common = ['how','to','i','to','the','z','na','do','is'
    ,'in','and','w','do','a','po','of','-','on','for'
       ,'o','–','co','czy']
for i in range(100):
    counter[str(i)] = 0
for i in common:
    counter[i] = 0
a = counter.most_common(40)
words = [x[0] for x in a]
count = [x[1] for x in a]
plt.plot(words,count)
plt.xticks(rotation = 90)
plt.show()


plt.hist(data['date'],bins = 100)
plt.xticks(rotation=90)
plt.show()


word = 'torrent'
plt.hist(data[(data['search'].map(lambda x: (x.lower()).split())).map(lambda x:x.count(word)>0)]['date'],bins = 50)
plt.xticks(rotation=90)
plt.title(word)
plt.show()


# plotowanie gestosci wyszukiwan na dzien i na godzine
plt.figure(figsize=(15, 4))
a = data.groupby([data['date'].map(lambda x:x.weekday()), data['date'].map(lambda x:x.hour)])['search'].count().unstack().fillna(0)
sns.heatmap(a, cmap='plasma')
plt.xlabel('hour')
plt.ylabel('day of a week')
plt.show()


word = 'torrent'
plt.figure(figsize=(15, 4))
a = data[(data['search'].map(lambda x: (x.lower()).split())).map(lambda x:x.count(word)>0)].groupby([data['date'].map(lambda x:x.weekday()), data['date'].map(lambda x:x.hour)])['search'].count().unstack().fillna(0)
sns.heatmap(a, cmap='plasma')
plt.title(word)
plt.xlabel('hour')
plt.ylabel('day of a week')
plt.show()



years = data['date'].map(lambda x: x.year).unique()
for year in years:

    plt.figure(figsize=(15, 4))
    a = data[data['date'].map(lambda x: x.year) == year].groupby([data['date'].map(lambda x:x.weekday()), data['date'].map(lambda x:x.hour)])['search'].count().unstack().fillna(0)
    sns.heatmap(a,cmap='plasma')
    plt.title(year)
    # plt.show()
    plt.savefig('piter'+str(year))
    plt.close()