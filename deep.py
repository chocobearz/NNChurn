import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

cd = pd.read_csv("cleanData.csv")

corr = cd.corr()

sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
sns.plt.show()