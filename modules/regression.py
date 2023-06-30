import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
import numpy as np
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

def linear_regression(x_train, y_train,x_test=[], y_test=[]):
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    if x_test == [] and y_test == []:
      x_test=x_train
      y_test=y_train
    y_pred = lr.predict(x_train)
    mae = metrics.mean_absolute_error(y_test,y_pred)
    mse = metrics.mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test,y_pred))
    return {"model":lr,"predict":y_pred,"score":lr.score(x_test, y_test),"error":{"mae":mae,"mse":mse,"rmse":rmse}}

def regularization_regression(x_train, y_train,alpha_start=0,alpha_end=0,alpha_step=1,algorithm='ridge',x_test=[], y_test=[],alpha=1.0,l1_ratio=0.5):
    alpha_intial=alpha    
    algorithms={'ridge':Ridge,'lasso':Lasso,'elasticnet':ElasticNet}
    score_initial=0.0    
    if x_test == [] and y_test == []:
      x_test=x_train
      y_test=y_train
    for alpha_ in np.arange(alpha_start,alpha_end,alpha_step):
      if(algorithm=='elasticnet'):
        rr = algorithms[algorithm](alpha=alpha_, l1_ratio=l1_ratio)  
      else:
        rr = algorithms[algorithm](alpha=alpha_)
      rr.fit(x_train, y_train)
      if rr.score(x_test, y_test)>score_initial:
        score_initial=rr.score(x_test, y_test)  
        alpha_intial=alpha_
    rr = Ridge(alpha=alpha_intial)
    rr.fit(x_train, y_train)
    y_pred = rr.predict(x_train)
    mae = metrics.mean_absolute_error(y_test,y_pred)
    mse = metrics.mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test,y_pred))
    return {"model":rr,"predict":y_pred,"score":rr.score(x_test, y_test),"error":{"mae":mae,"mse":mse,"rmse":rmse},"bestalpha":alpha_intial}

def logistic_regression(x_train, y_train,x_test=[], y_test=[],solver='liblinear', random_state=0):
    lr = LogisticRegression(solver=solver, random_state=random_state)
    lr.fit(x_train, y_train)
    if x_test == [] and y_test == []:
      x_test=x_train
      y_test=y_train
    y_pred = lr.predict(x_train)
    mae = metrics.mean_absolute_error(y_test,y_pred)
    mse = metrics.mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test,y_pred))
    return {"model":lr,"predict":y_pred,"score":lr.score(x_test, y_test),"error":{"mae":mae,"mse":mse,"rmse":rmse}}




##regularization_regression = regularization_regression(X,Y,algorithm='ridge',alpha_step=0.0001,alpha_end=1,alpha_start=0.01)
##regularization_regression = regularization_regression(X,Y,algorithm='ridge')

##print(regularization_regression['score'])
##print(regularization_regression['error'])
##print(regularization_regression['bestalpha'])
