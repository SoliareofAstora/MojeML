import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

input = pd.read_csv('input/airports.csv',usecols=[1,2,3,4,5,6,7,8,9,10,11],names=['name','city','country','IATA','ICAO','lat','lon','alt','tz','DST','tzolson'])

plt.scatter(input['lon'],  input['lat'],0.05,alpha=0.6)
plt.show()

input['country'].tail(12)

input.iloc[1]
input.loc[1]

input[input['country']=='Poland']['name']

input[input['city']!=input['name']][['name','city']]

input['alt'] = input['alt'].map(lambda x: x*30.48/100)

input[~ input['country'].duplicated(keep=False)]['country']

input = input[~ input['country'].duplicated(keep=False)]


air = pd.read_csv('input/airports.csv',usecols=[1,2,3,4,5,6,7,8,9,10,11],names=['name','city','country','IATA','ICAO','lat','lon','alt','tz','DST','tzolson'])
area = pd.read_csv('input/area.csv')

a = air['country'].unique()
a = np.append(a,area['Country Name'])
a,count = np.unique(a,return_counts=True)
ERRORS = a[count==1]

# zbiory danych sa nie kompatybilne. tabela area nie zawiera 4 panstw
