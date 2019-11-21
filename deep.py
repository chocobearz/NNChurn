import pandas as pd
import seaborn as sns

cd = pd.read_csv("cleanData.csv")

corr = cd.corr()

sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
sns.plt.show()