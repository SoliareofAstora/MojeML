import pandas as pd

data = pd.read_csv("input/spam.csv",encoding='latin-1')

# Drop columns and rename
data = data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
data = data.rename(columns={'v1':'label','v2':'test'})

data.sample(3)


