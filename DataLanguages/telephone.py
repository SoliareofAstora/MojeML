import pandas as pd

input = pd.read_csv('input/telephone_calls.csv')

data = pd.DataFrame()
data['name'] = input['Name1'].append(input['Name2'],ignore_index=True)
data['tel'] = input['Telephone1'].append(input['Telephone2'],ignore_index=True)

data['tel'].drop_duplicates(inplace=True)
