"""this code was informed from the following repsitory
https://github.com/shakedzy/dython"""

import pandas as pd
import scipy.stats as ss
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.tight_layout()
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

#correlation ratio to find association between numeric and categroical variable
def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator/denominator)
    return eta


oneHot = pd.read_csv("oneHotNNData.csv")


cd = pd.read_csv("cleanData.csv")
#build correlation matrix
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
          cell = round(correlation_ratio(cd[columns[i]], cd[columns[j]]),4)
          corr[columns[i]][columns[j]] = cell
          corr[columns[j]][columns[i]] = cell
      else:
        if columns[j] not in numeric_columns:
          cell = round(correlation_ratio(cd[columns[j]], cd[columns[i]]),4)
          corr[columns[i]][columns[j]] = cell
          corr[columns[j]][columns[i]] = cell
        else:
          cell = round((ss.pearsonr(cd[columns[i]], cd[columns[j]])[0]), 4)
          corr[columns[i]][columns[j]] = cell
          corr[columns[j]][columns[i]] = cell

corr = corr[corr.columns].astype(float)

#plot correlation matrix
sns.heatmap(
  corr, 
  xticklabels=corr.columns.values,
  yticklabels=corr.columns.values
)

plt.savefig('corr.png')

plt.tight_layout()
corr2 = oneHot.corr()

sns.heatmap(
  corr2, 
  xticklabels=corr2.columns.values,
  yticklabels=corr2.columns.values
)

plt.savefig('corrHot.png')


