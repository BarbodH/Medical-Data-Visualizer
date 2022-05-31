import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('medical_examination.csv')

# add 'overweight' column
bmi = df['weight'] / (df['height'] / 100) ** 2
df['overweight'] = (bmi > 25).astype(int) # astype(int) casts series from boolean to integer

# normalize data by making 0 always good and 1 always bad
# if the value of 'cholesterol' or 'gluc' is 1, make the value 0
# if the value is more than 1, make the value 1.
df.loc[df['cholesterol'] == 1, 'cholesterol'] = 0
df.loc[df['cholesterol'] > 1, 'cholesterol'] = 1
df.loc[df['gluc'] == 1, 'gluc'] = 0
df.loc[df['gluc'] > 1, 'gluc'] = 1

# draw Categorical Plot
def draw_cat_plot():
  df_cat = df.melt(id_vars='cardio', value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
  df_cat['total'] = 0
  df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).count()

  # draw & save catplot using Seaborn
  fig = sns.catplot(x='variable', y='total', hue='value', kind='bar', col='cardio', data=df_cat).fig
  fig.savefig('catplot.png')
  return fig


# draw heat map
def draw_heat_map():
  # clean the data: remove 'heihgt' & 'weight' values higher than 97.5th & lower than 2.5th percentile
  df_heat = df[(df['ap_lo'] <= df['ap_hi'])
    & (df['height'] >= df['height'].quantile(0.025)) 
    & (df['height'] <= df['height'].quantile(0.975)) 
    & (df['weight'] >= df['weight'].quantile(0.025)) 
    & (df['weight'] <= df['weight'].quantile(0.975))]

  # calculate the correlation matrix & generate mask for upper triangle
  corr_matrix = df_heat.corr() # method is 'pearson' by default
  mask = np.triu(corr_matrix)

  # draw & save heatmap using Seaborn
  fig, ax = plt.subplots()
  sns.heatmap(data=corr_matrix, annot=True, square=True, mask=mask, fmt='.1f', center=0.08, cbar_kws={'shrink': 0.5})
  fig.savefig('heatmap.png')
  return fig
