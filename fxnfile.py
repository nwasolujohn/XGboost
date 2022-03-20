import numpy as np
import seaborn as sns
import scipy.integrate as it
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import GridSearchCV as gs


# A function that gets the mean squared percentage  error

def mape (observed, predicted):
    b = [0]*len(observed)
    x, y = np.array(observed), np.array(predicted)
    for i in range (len(y)):
        if x[i] != 0:   #this is used to prevent 0 division
            b[i] = np.abs((x[i]-y[i])/x[i])
        else:
            b[i] = y[i]/np.mean(x)
    return np.mean(b)*100


# Evaluation function
    
def evaluation (observed, predicted):
    print('MAE: ', mae(observed, predicted))
    print('MSE: ', mse(observed, predicted))
    print('MAPE: ', mape(observed, predicted)) 
    print('RMSE: ', np.sqrt(mse(observed, predicted)))
    
# Emission rate 
    
def emission_rate_WLTC (observed, time):
    # Printitng the emission rate at every 100 mili seconds is too noisy
    # The following code converts series to array and cleans the noise by taking the mean of every nth value
    time_s = np.mean(time.values[:(len(time)//40)*40].reshape(-1, 40), axis = 1)
    obs = np.mean(observed.values[:(len(observed)//40)*40].reshape(-1, 40), axis = 1)
    plt.figure(figsize = (20, 12))
    gas = input("emission type, CO, CO2, Nox or HC: ")
    if (gas == 'CO'):
        plt.ylabel('CO (avaraged over 4 seconds)', fontsize = 20)
    elif (gas == 'CO2'):
        plt.ylabel('CO2 (avaraged over 4 seconds)', fontsize = 20)
    elif (gas == 'Nox'):
        plt.ylabel('Nox (avaraged over 4 seconds)', fontsize = 20)
    else:
        plt.ylabel('HC (avaraged over 4 seconds)', fontsize = 20)
    plt.xlabel('Time (avaraged over 4 seconds)', fontsize = 20)
    temp = input ('starting temp, 0c or 85c: ')
    if (temp == '0c'):
        plt.title('Emission Rate 0c WLTC', fontsize = 20)
    else:
        plt.title('Emission Rate 85c WLTC', fontsize = 20)
    plt.plot(time_s, obs)
    plt.show()
    
    
# Comparing the emission rate of predicted and observed
    
def compare_emi_rate_WLTC (observed, predicted, time):
    time_s = np.mean(time.values[:(len(time)//40)*40].reshape(-1, 40), axis = 1)
    obs = np.mean(observed.values[:(len(observed)//40)*40].reshape(-1, 40), axis = 1)
    pred = np.mean(predicted[:(len(predicted)//40)*40].reshape(-1, 40), axis = 1)
    #plt.figure(figsize = (20,12))
    gas = input("emission type, CO, CO2, Nox or HC: ")
    if (gas == 'CO'):
        plt.ylabel('CO (avaraged over 4 seconds)', fontsize = 20)
    elif (gas == 'CO2'):
        plt.ylabel('CO2 (avaraged over 4 seconds)', fontsize = 20)
    elif (gas == 'Nox'):
        plt.ylabel('Nox (avaraged over 4 seconds)', fontsize = 20)
    else:
        plt.ylabel('HC (avaraged over 4 seconds)', fontsize = 20)
    plt.xlabel('Time (avaraged over  seconds)', fontsize = 20)
    ax = plt.subplot()
    ax.plot(time_s, obs, label = 'Observed', color = 'orange')
    ax.plot(time_s, pred, label = 'Predicted', linestyle = '--')
    temp = input ('starting temp, 0c or 85c: ')
    if (temp == '0c'):
        plt.title('Emission Rate 0c WLTC', fontsize = 20)
    else:
        plt.title('Emission Rate 85c WLTC', fontsize = 20)
    ax.legend()
    plt.show()
    


# This function gives the predicted values 

def prediction (variables_train_x, emission_train_y, variable_test_x, best_parameter):
    clf = XGBRegressor(best_parameter)
    clf.fit(train_x_variables, train_y_variable)
    pred = clf.predict(test_x_variables)
    return pred

# This function compares the emission rata with time

def integral_plot_WLTC (observed, predicted, time):
    #plt.figure(figsize = (20, 12))
    gas = input("emission type, CO, CO2, Nox or HC: ")
    if (gas == 'CO'):
        plt.ylabel('CO ', fontsize = 20)
    elif (gas == 'CO2'):
        plt.ylabel('CO2 ', fontsize = 20)
    elif (gas == 'Nox'):
        plt.ylabel('Nox ', fontsize = 20)
    else:
        plt.ylabel('HC ', fontsize = 20)
    plt.xlabel('Time', fontsize = 20)
    ax = plt.subplot()
    ax.plot(time[:-1], it.cumtrapz(observed, x=time), label = 'Observed', color = 'orange')
    ax.plot(time[:-1], it.cumtrapz(predicted, x=time), label = 'Predicted', linestyle = '--')
    temp = input ('starting temp, 0c or 85c: ')
    if (temp == '0c'):
        plt.title('Emission Rate 0c WLTC', fontsize = 20)
    else:
        plt.title('Emission Rate 85c WLTC', fontsize = 20)
    ax.legend()
    plt.show()
