import pandas as pd

data_dir = 'input/'
df_seeds = pd.read_csv(data_dir + 'WNCAATourneySeeds.csv')
df_tour = pd.read_csv(data_dir + 'WNCAATourneyCompactResults.csv')

# df_seeds.sample(10)
# df_tour.sample(10)


