from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt

def feature_selection_prediction(X_train,y_train):
  rf = RandomForestRegressor(random_state=0)
  rf.fit(X_train,y_train)
  f_i = list(zip(X_train.columns,rf.feature_importances_))
  f_i.sort(key = lambda x : x[1])
  plt.barh([x[0] for x in f_i],[x[1] for x in f_i])
  plt.show()

