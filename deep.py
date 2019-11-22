"""this code was informed from the following repsitory
https://github.com/shakedzy/dython"""

import pandas as pd
import scipy.stats as ss
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#cramersV to find association between categorical variables
def cramers_v(x, y):
  confusion_matrix = pd.crosstab(x,y)
  chi2 = ss.chi2_contingency(confusion_matrix)[0]
  n = confusion_matrix.sum().sum()
  phi2 = chi2/n
  r,k = confusion_matrix.shape
  phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
  rcorr = r-((r-1)**2)/(n-1)
  kcorr = k-((k-1)**2)/(n-1)
  return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

cd = pd.read_csv("cleanData.csv")

columns = cd.columns
numeric_columns = ["tenure", "monthlycharges", "totalcharges"]
corr = pd.DataFrame(index=columns, columns=columns)
for i in range(0,len(columns)):
  for j in range(i,len(columns)):
    if i == j:
        corr[columns[i]][columns[j]] = 1.0
    else:
      if columns[i] not in numeric_columns:
        if columns[j] not in numeric_columns:
          cell = round(cramers_v(cd[columns[i]], cd[columns[j]]), 4)
          corr[columns[i]][columns[j]] = cell
          corr[columns[j]][columns[i]] = cell
        else:
          cell = round((ss.f_oneway(cd[columns[i]], cd[columns[j]])[1]),4)
          corr[columns[i]][columns[j]] = cell
          corr[columns[j]][columns[i]] = cell
      else:
        if columns[j] not in numeric_columns:
          cell = round((ss.f_oneway(cd[columns[j]], cd[columns[i]])[1]),4)
          corr[columns[i]][columns[j]] = cell
          corr[columns[j]][columns[i]] = cell
        else:
          cell = round((ss.pearsonr(cd[columns[i]], cd[columns[j]])[0]), 4)
          corr[columns[i]][columns[j]] = cell
          corr[columns[j]][columns[i]] = cell

corr = corr[corr.columns].astype(float)

sns.heatmap(
  corr, 
  xticklabels=corr.columns.values,
  yticklabels=corr.columns.values
)

plt.savefig('corr2.png')



