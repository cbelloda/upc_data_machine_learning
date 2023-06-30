#import seaborn as sns
#import matplotlib.pyplot as plt
#sns.pairplot(df[info['numeric_columns']])
#plt.show()



#!pip install fitter
#import seaborn as sns
#from fitter import Fitter, get_common_distributions, get_distributions
#info['numeric_columns']
#sns.set_style('white')
#sns.set_context("paper", font_scale = 2)
#sns.displot(data=df, x="Salary", kind="hist", bins = 100, aspect = 1.5)
#salary = df["Salary"].values
#from fitter import get_common_distributions
#get_common_distributions()
#f = Fitter(salary,
#           distributions=get_common_distributions())
#f.fit()
#f.summary()
#f.get_best(method = 'sumsquare_error')