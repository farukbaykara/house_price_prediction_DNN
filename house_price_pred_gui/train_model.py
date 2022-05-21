#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow.python.client import device_lib


import itertools



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import matplotlib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import IsolationForest

import warnings

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from keras_tuner import HyperModel
from keras_tuner.tuners import RandomSearch, Hyperband
import IPython
from tensorflow.keras.metrics import Metric 
from keras_tuner import HyperModel
from keras_tuner.tuners import Hyperband
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "99"
device_lib.list_local_devices()
import sys
sys.path.insert(0, '..')

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.python.keras.optimizers import adam_v2





train_data = pd.read_csv('/home/baykara/Kaggle/house_prices_prediction/data_set/train.csv')
test_data = pd.read_csv('/home/baykara/Kaggle/house_prices_prediction/data_set/test.csv')
train_data.head()

# 1. Excluding columns which have majority as null
train_data= train_data.loc[:, train_data.isnull().sum()/len(train_data)<0.80]

x= train_data.iloc[:, 1:-1] # Dropping 'Id' and the Y feature
y= train_data.iloc[:,-1]

train_cols= x.columns


# 2. Looking at the Overall statistics of variables and correlation among all variables
train_stats= x.describe().transpose()
train_stats

ordinal_cols= list(x.columns[x.columns.str.contains('Yr|Year')])
#print('ordinal/temporal columns:\n',ordinal_cols)
nominal_cols= list(set(x.select_dtypes(include=['object']).columns)- set(ordinal_cols))
#print('nominal columns:\n', nominal_cols)
numeric_cols= list(set(x.select_dtypes(exclude=['object']).columns)- set(ordinal_cols))
#print('numeric columns:\n',numeric_cols)


x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.20, random_state=0)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


def missing_val_imputation(x, ordinal_cols,nominal_cols,numeric_cols):
    
    for col in ordinal_cols:
        x.loc[:,col]= x.loc[:,col].fillna(x.loc[:,col].mean())

    x.loc[:,nominal_cols]= x.loc[:,nominal_cols].fillna("?")
    
    for col in numeric_cols:
        x.loc[:,col]= x.loc[:,col].fillna(x.loc[:,col].mean())
#         x.loc[:,col]= x.groupby("OverallQual")[col].transform(lambda grp:grp.fillna(np.mean(grp)))

    print("All missing values are now imputed!\n",x.isnull().sum().sort_values(ascending=False))
    
    return x


x_train= missing_val_imputation(x_train,ordinal_cols,nominal_cols,numeric_cols)
x_test= missing_val_imputation(x_test,ordinal_cols,nominal_cols,numeric_cols)



# Fitting OHE object
ohe= OneHotEncoder(handle_unknown='ignore', sparse=False).fit(x_train[nominal_cols]) 

#Feature Encoding for nominal columns

def ohe_transform(x, ohe, nominal_cols):
    x_ohe= pd.DataFrame(ohe.transform(x[nominal_cols]))
    x_ohe.columns=ohe.get_feature_names(nominal_cols)

    # prepping x
    x=x.drop(nominal_cols, axis=1)
    x.reset_index(inplace=True, drop=True)
    x= x.merge(x_ohe, left_index=True, right_index=True)
    
    return x

x_train= ohe_transform(x_train, ohe, nominal_cols)
x_test= ohe_transform(x_test, ohe, nominal_cols)
print(x_train.shape, x_test.shape)



# Standard Scaling
ss= StandardScaler()
x_train_ss= pd.DataFrame(ss.fit_transform(x_train))
x_test_ss= pd.DataFrame(ss.transform(x_test))




sel= SelectFromModel(Lasso(alpha=0.5, max_iter=3000, tol=0.005, random_state=0, warm_start= False)) # warm_start= True

# train the lasso model and select features
sel.fit(x_train_ss, y_train)

sel.get_support()

#selected_feats= x_train_ss.columns[(sel.get_support())]

selected_feats = [0,1,2,3,4,5,6,7,8,9]

# print the stats
print("# of total features: ",x_train.shape[1])
print("# of selected features: ",len(selected_feats))
# print("# of rejected features: ",np.sum(sel.estimator_.coef_==0))
# print('Selected features:', selected_feats)

x_train_ss= x_train_ss[selected_feats]
x_test_ss= x_test_ss[selected_feats]

# Building a neural network
import tensorflow.keras

optimizer= tf.keras.optimizers.Adam(0.001)

normal_model = Sequential()
normal_model.add(Dense(128, input_shape=[len(x_train_ss.keys())], kernel_initializer='normal', activation='relu'))
normal_model.add(Dense(64, kernel_initializer='normal', activation='relu'))
normal_model.add(Dense(32, kernel_initializer='normal', activation='relu'))
normal_model.add(Dense(1, kernel_initializer='normal'))
normal_model.compile(loss='mse', optimizer = optimizer, metrics=['mean_absolute_error','mean_squared_error'])

# Build and inspect the model



early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min', restore_best_weights=False)

history= normal_model.fit(
    x_train_ss, y_train,
    epochs=200,
    validation_data=(x_test_ss, y_test),
    verbose=0, #set verbose=1 for full details at every epoch
    callbacks= [early_stopping_cb])

loss, mae, mse= normal_model.evaluate(x_test_ss, y_test, verbose=2)

print("Test-set Mean absolute error: {:5.2f}".format(mae)) # test mae- 36286


y_pred_test= normal_model.predict(x_test_ss).flatten()

error= y_pred_test-y_test
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [SalesPrice]')
_=plt.ylabel('Count')

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error



y_pred_train= normal_model.predict(x_train_ss).flatten()

print("Accuracy obtained using x_train and x_val sets from the original x!")

print("Training accuracy: ",r2_score(y_train, y_pred_train))

print("Test accuracy: ",r2_score(y_test, y_pred_test))

print("Test mean-squared error: ",np.sqrt(mean_squared_error(y_test, y_pred_test)))





class ANNhypermodel(HyperModel):
    
    def __init__(self, input_shape):
        self.input_shape= input_shape
        
    def build(self, hp):
        model= tf.keras.Sequential()
        
        # Tune the number of units in the first Dense layer
        # Defining dense units as a close approx to the original neural network to perform a fair comparision!
        
        
        hp_units_1= hp.Int('units_1', min_value=128, max_value= 160, step=32)
        hp_units_2= hp.Int('units_2', min_value=64, max_value= 128, step=32)
        hp_units_3= hp.Int('units_3', min_value=32, max_value= 64, step=16)

        model.add(tf.keras.layers.Dense(units=hp_units_1, activation='relu', input_shape= self.input_shape))
        model.add(tf.keras.layers.Dense(units=hp_units_2, activation='relu'))
        model.add(tf.keras.layers.Dense(units=hp_units_3, activation='relu'))
        model.add(tf.keras.layers.Dense(1))

        # Tune the learning rate for the optimizer 
        hp_learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default= 0.0005)

        model.compile(loss='mse',
                    optimizer= tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                    metrics= ['mae','mse']
                    )

        return model

hypermodel= ANNhypermodel(input_shape= [len(x_train_ss.keys())])

HYPERBAND_MAX_EPOCHS = 150
EXECUTION_PER_TRIAL = 2

tuner= Hyperband(hypermodel,
                objective= 'val_mse',
                max_epochs=HYPERBAND_MAX_EPOCHS, #Set 100+ for good results
                executions_per_trial=EXECUTION_PER_TRIAL,
                directory= 'hyperband',
                project_name='houseprices',
                overwrite=True)

from time import time

print('searching for the best params!')

t0= time()
tuner.search(x= x_train_ss,
            y= y_train,
            epochs=100,
            batch_size= 64,
            validation_data= (x_test_ss, y_test),
            verbose=0,
            callbacks= []
            )
print(time()- t0," secs")

# Retreive the optimal hyperparameters
best_hps= tuner.get_best_hyperparameters(num_trials=1)[0]

# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the 
first densely-connected layer is {best_hps.get('units_1')},
second layer is {best_hps.get('units_2')} 
third layer is {best_hps.get('units_3')}  

and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# Evaluate the best model.
print(best_model.metrics_names)
loss, mae, mse = best_model.evaluate(x_test_ss, y_test)
print(f'loss:{loss} mae: {mae} mse: {mse}')
# Build the model with the optimal hyperparameters and train it on the data
tuned_model = tuner.hypermodel.build(best_hps)

# Check result using best model
t00= time()
history_tuned= tuned_model.fit(x_train_ss, y_train, 
        epochs = 200, 
        validation_data = (x_test_ss, y_test),
        verbose=0,
        callbacks= early_stopping_cb)

# print(time()- t00," secs")

print("\n Using Early stopping, needed only ",len(history_tuned.history['val_mse']),"epochs to converge!")



y_pred_train_tuned= tuned_model.predict(x_train_ss).flatten()
y_pred_test_tuned= tuned_model.predict(x_test_ss).flatten()

print("Training accuracy: ",r2_score(y_train, y_pred_train_tuned))

print("Test accuracy: ",r2_score(y_test, y_pred_test_tuned))

print("Test mean-squared error: ",np.sqrt(mean_squared_error(y_test, y_pred_test_tuned)))

    
normal_model.save("normal_model.h5")
tuned_model.save("tuned_model.h5")

print("Models save!!!")







