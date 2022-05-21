#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
#from tensorflow.python.client import device_lib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams


from sklearn.model_selection import train_test_split




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


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



from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.models import load_model


def missing_val_imputation(x, ordinal_cols,nominal_cols,numeric_cols):
    
    for col in ordinal_cols:
        x.loc[:,col]= x.loc[:,col].fillna(x.loc[:,col].mean())

    x.loc[:,nominal_cols]= x.loc[:,nominal_cols].fillna("?")
    
    for col in numeric_cols:
        x.loc[:,col]= x.loc[:,col].fillna(x.loc[:,col].mean())
#         x.loc[:,col]= x.groupby("OverallQual")[col].transform(lambda grp:grp.fillna(np.mean(grp)))

    #print("All missing values are now imputed!\n",x.isnull().sum().sort_values(ascending=False))
    
    return x


def ohe_transform(x, ohe, nominal_cols):
    x_ohe= pd.DataFrame(ohe.transform(x[nominal_cols]))
    x_ohe.columns=ohe.get_feature_names(nominal_cols)

    # prepping x
    x=x.drop(nominal_cols, axis=1)
    x.reset_index(inplace=True, drop=True)
    x= x.merge(x_ohe, left_index=True, right_index=True)
    
    return x



def predict_price(user_input,model_selection):
    train_data = pd.read_csv('/home/baykara/Kaggle/house_prices_prediction/data_set/train.csv')

    test_data = pd.read_csv('/home/baykara/Kaggle/house_prices_prediction/data_set/test.csv')
    
    x= train_data.iloc[:, 1:-1] # Dropping 'Id' and the Y feature
    y= train_data.iloc[:,-1]

    train_cols = x.columns
    
    test_data = test_data[train_cols]
    
    ordinal_cols= list(x.columns[x.columns.str.contains('Yr|Year')])

    nominal_cols= list(set(x.select_dtypes(include=['object']).columns)- set(ordinal_cols))

    numeric_cols= list(set(x.select_dtypes(exclude=['object']).columns)- set(ordinal_cols))

    x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.20, random_state=0)

    x_train= missing_val_imputation(x_train,ordinal_cols,nominal_cols,numeric_cols)
    x_test= missing_val_imputation(x_test,ordinal_cols,nominal_cols,numeric_cols)

    ohe= OneHotEncoder(handle_unknown='ignore', sparse=False).fit(x_train[nominal_cols]) 

    #Finding selected features
    x_train= ohe_transform(x_train, ohe, nominal_cols)
    x_test= ohe_transform(x_test, ohe, nominal_cols)
    ss= StandardScaler()
    x_train_ss= pd.DataFrame(ss.fit_transform(x_train))
    x_test_ss= pd.DataFrame(ss.transform(x_test))

    sel= SelectFromModel(Lasso(alpha=0.5, max_iter=3000, tol=0.005, random_state=0, warm_start= False)) # warm_start= True

    # train the lasso model and select features
    sel.fit(x_train_ss, y_train)

    sel.get_support()

    #selected_feats= x_train_ss.columns[(sel.get_support())]

    selected_feats = [0,1,2,3,4,5,6,7,8,9]
    
    samp_test_data= ohe_transform(test_data, ohe, nominal_cols)

    samp_test_data.loc[0,'MSSubClass'] = user_input[0]
    samp_test_data.loc[0,'LotFrontage'] = user_input[1]
    samp_test_data.loc[0,'LotArea'] = user_input[2]
    samp_test_data.loc[0,'OverallQual'] = user_input[3]
    samp_test_data.loc[0,'OverallCond'] = user_input[4]
    samp_test_data.loc[0,'YearBuilt'] = user_input[5]
    samp_test_data.loc[0,'YearRemodAdd'] = user_input[6]
    samp_test_data.loc[0,'MasVnrArea'] = user_input[7]
    samp_test_data.loc[0,'BsmtFinSF1'] = user_input[8]
    samp_test_data.loc[0,'BsmtFinSF2'] = user_input[9]

    test_data_ss= pd.DataFrame(ss.transform(samp_test_data))

    test_data_ss= test_data_ss[selected_feats]

    normal_model = load_model("normal_model.h5")
    tuned_model = load_model("tuned_model.h5")
    if(model_selection ==1):
        user_predictions= tuned_model.predict(test_data_ss).flatten()
    elif(model_selection ==0):
        user_predictions= normal_model.predict(test_data_ss).flatten()
    

    return user_predictions[0]

