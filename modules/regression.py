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

def regularization_regression(x_train, y_train,algorithm='ridge',x_test=[], y_test=[],alpha=1.0,alpha_set=[]):
    alpha_intial=alpha    
    algorithms={'ridge':Ridge,'lasso':Lasso,'elasticnet':ElasticNet}
    for alpha_ in alpha_set:
      rr = algorithms[algorithm](alpha=alpha_)
      rr.fit(x_train, y_train)
      if rr.score(x_test, y_test)>alpha_intial:
        alpha_intial=rr.score(x_test, y_test)        
    rr = Ridge(alpha=alpha_intial)
    rr.fit(x_train, y_train)
    if x_test == [] and y_test == []:
      x_test=x_train
      y_test=y_train
    
    y_pred = rr.predict(x_train)
    mae = metrics.mean_absolute_error(y_test,y_pred)
    mse = metrics.mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test,y_pred))
    return {"model":rr,"predict":y_pred,"score":rr.score(x_test, y_test),"error":{"mae":mae,"mse":mse,"rmse":rmse},"bestalpha":alpha_intial}


   
   
   



#lr = LogisticRegression(solver='liblinear', random_state=0)
#lr.fit(X,Y)
#Y_predict_lr = lr.predict_proba(X)