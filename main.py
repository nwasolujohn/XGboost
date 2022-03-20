import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV as gs

df = pd.ExcelFile('comb.xlsx').parse('0c')
corr = df.corr()
#
# Generate a mask for upper traingle
#
mask = np.triu(np.ones_like(corr, dtype=bool))
#
# Configure a custom diverging colormap
#
cmap = sns.diverging_palette(230, 20, as_cmap=True)
#
# Draw the heatmap
#
sns.heatmap(corr, annot=True, mask = mask, cmap=cmap)
plt.show()
#
#With the variable to use for the model can be decided
#
]# chosen test varaibles
variable_test_x = df[['N', 'Gear', 'Vxx', 'TCOOL', 'P_int', 'N_t', 'P_ext', 'VSP']
#
# promts for the emission to predict
#
emission_test_y = str(input('emission type, CO, CO2, HC, or Nox: ')) 
observed = df[emission_test_y]
time = df['Time_s']
#
df1 = pd.ExcelFile('comb.xlsx').parse('25c') 
variables_train_x = df1[['N', 'Gear', 'Vxx', 'TCOOL', 'P_int', 'N_t', 'P_ext', 'VSP']]
emission_train_y = df1[emission]                     
#                     
#Using gride search to tune hyperparameters
#                   
parameters = {'max_depth': [20, 19, 18, 17, 16, 15],
              'n_estimators':[900, 800, 700, 600, 500], 
              'learning_rate': [0.05, 0.04, 0.03, 0.02], 
              'subsample': [1, 0.8, 0.9]
              "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
              "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }}

estimator = XGBRegressor() 
grid_search = gs(estimator=estimator,param_grid=parameters,scoring = 'neg_root_mean_squared_error',cv = 10,verbose=True)
grid_search.fit(variables_train_x, emission_train_y)                      
best_parameter = grid_search.best_estimator_

predicted = best_parameter.predict(variable_test_x)
