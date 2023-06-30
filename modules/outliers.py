import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import scipy.stats as stats

def out_std(s, nstd=3.0, return_thresholds=False):
    data_mean, data_std = s.mean(), s.std()
    cut_off = data_std * nstd
    lower, upper = data_mean - cut_off, data_mean + cut_off
    if return_thresholds:
        return lower, upper
    else:
        return [True if x < lower or x > upper else False for x in s]

def out_iqr(s, k=1.5, return_thresholds=False):
    q25, q75 = np.percentile(s, 25), np.percentile(s, 75)
    iqr = q75 - q25
    cut_off = iqr * k
    lower, upper = q25 - cut_off, q75 + cut_off
    if return_thresholds:
        return lower, upper
    else: # identify outliers
        return [True if x < lower or x > upper else False for x in s]
    
def out_zscore(df):
    z = np.abs(stats.zscore(df))
    #data_clean = data[(z<3).all(axis=1)]    
    #data_clean.shape
    return z

def out_multi_lof(df):
  clf = LocalOutlierFactor(n_neighbors=10, contamination=0.1)
  X=df
  y_pred = clf.fit_predict(X)
  X_scores = clf.negative_outlier_factor_
  plt.title("Local Outlier Factor (LOF)")
  plt.scatter(X.iloc[:,0], X.iloc[:,1], color="k", s=3.0, label="Data points")
  radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
  plt.scatter(
      X.iloc[:,0],
      X.iloc[:,1],
      s=1000 * radius,
      edgecolors="r",
      facecolors="none",
      label="Outlier scores",
  )
  plt.show()  
  return X