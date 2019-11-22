import pandas as pd
import scipy.stats as ss
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def convert(data, to):
  converted = None
  if to == 'array':
    if isinstance(data, np.ndarray):
        converted = data
    elif isinstance(data, pd.Series):
        converted = data.values
    elif isinstance(data, list):
        converted = np.array(data)
    elif isinstance(data, pd.DataFrame):
        converted = data.as_matrix()
  elif to == 'list':
    if isinstance(data, list):
        converted = data
    elif isinstance(data, pd.Series):
        converted = data.values.tolist()
    elif isinstance(data, np.ndarray):
        converted = data.tolist()
  elif to == 'dataframe':
    if isinstance(data, pd.DataFrame):
        converted = data
    elif isinstance(data, np.ndarray):
        converted = pd.DataFrame(data)
  else:
      raise ValueError("Unknown data conversion: {}".format(to))
  if converted is None:
      raise TypeError('cannot handle data conversion of type: {} to {}'.format(type(data),to))
  else:
      return converted

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

def correlation_ratio(categories, measurements,):
  categories = convert(categories, 'array')
  measurements = convert(measurements, 'array')
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
          cell = cramers_v(cd[columns[i]], cd[columns[j]])
          corr[columns[i]][columns[j]] = cell
          corr[columns[j]][columns[i]] = cell
        else:
          cell = correlation_ratio(cd[columns[i]], cd[columns[j]])
          corr[columns[i]][columns[j]] = cell
          corr[columns[j]][columns[i]] = cell
      else:
        if columns[j] not in numeric_columns:
          cell = correlation_ratio(cd[columns[j]], cd[columns[i]])
          corr[columns[i]][columns[j]] = cell
          corr[columns[j]][columns[i]] = cell
        else:
          cell, _ = ss.pearsonr(cd[columns[i]], cd[columns[j]])
          corr[columns[i]][columns[j]] = cell
          corr[columns[j]][columns[i]] = cell
sns.heatmap(
  corr, 
  xticklabels=corr.columns.values,
  yticklabels=corr.columns.values
)

plt.savefig('corr.png')



