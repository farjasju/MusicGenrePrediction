import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('data/data.csv')

print(data.columns)
print(data.cov())
print(data.describe())

#Geting the correlation matrix
f = plt.figure(figsize=(19, 15))
plt.matshow(df.corr(), fignum=f.number)
plt.xticks(range(df.shape[1]), df.columns, fontsize=10, rotation=90)
plt.yticks(range(df.shape[1]), df.columns, fontsize=10)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=12)