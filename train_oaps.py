#!/usr/bin/env python3
# coding: utf-8
# ---------------------------------------------------------------
# Glucose prediction script for OAPS data using LSTM.
# Author: MedLaunch SugarBuddy
# ---------------------------------------------------------------

#datastructure packages
import numpy as np
import pandas as pd

#file system packages
import sys
import os
from os import path
import glob
import warnings
warnings.filterwarnings('ignore')
import gzip

#datetime packages
import datetime
import time
import dateutil.parser
import pytz

#machine learning packages
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

#miscellaneous
import math
from joblib import dump, load
import random
import pickle5 as pickle

# Specify GPU Here

#(default configuration for model)
state_vector_length = 32 #after manual grid search
epochs = 100
batch_size = 248
activation = 'relu' #activation for LSTM and RNN

# unpickle a pickled dictionary
def unpickle_data(data_path):
    with open(data_path, 'rb') as f:
        unpickled_data = pickle.load(f, encoding='latin1')
    return unpickled_data

# re-constructs data based on single-step/multi-output settings
# returns X, y as "features" (historical data) and "labels" (future data)
def process_data(df):
    excluded_keys = list()
    for key in df.keys():
        if 'date' in key:
            excluded_keys.append(key)
    df.drop(excluded_keys, axis=1, inplace=True)
    
    data = df.values
    data = data.astype('float32')

    # Multi prediction
    X, y = data[:, :-prediction_window], data[:, -prediction_window:] #x(t), x(t+1), ... , x(t+5)
        
    return X , y


def deepLSTMModel(model_name,X,y):
    '''Initializes deep learning model with LSTM.'''
    model = Sequential()

    # Define layers here:
    model.add(LSTM(state_vector_length, activation='relu', input_shape=(X.shape[1], X.shape[2])))
    prediction_window = y.shape[1]

    model.add(Dense(prediction_window, activation='relu'))
    # TODO: look into dropout layer

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    # TODO: can define learning rate in optimizer, try out diff loss functions

    return model

#******************************* MAIN Function for CGM prediction ***************************************
def CGM_Prediction(model_name,unpickled_train_data ,unpickled_test_data,ij ):

    train_subjs = list(unpickled_train_data.keys())
    test_subjs = list(unpickled_test_data.keys())
    random.shuffle(train_subjs)

    #nested function to test the model trained below
    def test_model(model,df):
        test_X, test_y = process_data(df) #get features and labels for test data
        
        #[no. of samples, timestamps, no. of features]
        test_X = test_X.reshape((test_X.shape[0], history_window , n_features))
        
        y_bar = model.predict(test_X)
    
        if prediction_type == 'multi':
            for ii in range(len(y_bar)):
                y_bar[ii] = [int(element) for element in y_bar[ii]]
        else:
            y_bar = [int(element) for element in y_bar]

        predicted_values = pd.DataFrame(list(zip(test_y,y_bar)),columns=['True','Estimated'])

        #for multi-output forecasting, calculate RMSE using the last value in the sequence
        if prediction_type == 'multi':
            y_bar = [last for *_, last in y_bar]
            test_y = [last for *_, last in test_y]
        
        testScore = math.sqrt(mean_squared_error(y_bar, test_y))
        return testScore, predicted_values

    def train_model(model,train_X,train_y):
        n_train = int(0.7*train_X.shape[0])
        
        train_X, val_X = train_X[:n_train, :,:], train_X[n_train:, :,:]
        train_y, val_y = train_y[:n_train], train_y[n_train:]
        early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
        history = model.fit(train_X, train_y, validation_data=(val_X, val_y), shuffle=False, epochs=epochs,batch_size=batch_size,verbose=0,callbacks=[early_stop])
        y_bar = model.predict(val_X)
        
        
        if prediction_type == 'multi':
            for ii in range(len(y_bar)):
                y_bar[ii] = [int(element) for element in y_bar[ii]]
        else:
            y_bar = [int(element) for element in y_bar]

        #for multi-output forecasting, calculate RMSE using the last value in the sequence
        if prediction_type == 'multi':
            y_bar = [last for *_, last in y_bar]
            val_y = [last for *_, last in val_y]

        valScore = math.sqrt(mean_squared_error(y_bar, val_y))
        print('Validation RMSE: %.3f' % valScore) 
        return model, history, valScore

    testScores = list()
    valScores = list() #validation error
    subjects = list()
    model_train_history = list()
    model_val_history = list()
    counter = 0 #keeping track of when the model was initialized

    for subj in train_subjs:
        print('----------Training on subject: ',subj,'----------')
        df = unpickled_train_data[subj].copy()
        train_X, train_y = process_data(df)

        #[no. of samples, timestamps, no. of features]
        
        train_X = train_X.reshape((train_X.shape[0], history_window , n_features))

        if counter == 0:
            model = deepLSTMModel(model_name,train_X, train_y)
        counter = counter + 1

        model,history,valScore = train_model(model,train_X,train_y)

        
        model_train_history.extend(history.history['mean_squared_error'])
        model_val_history.extend(history.history['mean_squared_error'])

        valScores.append(valScore)
        subjects.append(subj)

    all_subjs_predicted_values = {}
    for subj in subjects:
        print('----------Testing on subject: ',subj,'----------')
        df = unpickled_test_data[subj].copy()
        testScore, predicted_values = test_model(model,df)
        all_subjs_predicted_values[subj] = predicted_values 
        print('Test RMSE: %.3f' % testScore) 
        testScores.append(testScore)              

    results_df = pd.DataFrame(list(zip(subjects,valScores,testScores)),columns=['Subject','valRMSE','testRMSE'])
    results_df.sort_values(by=['Subject'], inplace = True)      
    return results_df, model, model_train_history,model_val_history,all_subjs_predicted_values

def make_directories():
    #datetime_now = datetime.datetime.now()
    #datetime_now = datetime_now.strftime("%d-%m-%Y_%I-%M-%S_%p")
    if not path.exists(output_directory):
        os.mkdir(output_directory)
    if not path.exists(output_directory + 'overall_results'):
        os.mkdir(output_directory + 'overall_results')
    if not path.exists(output_directory + 'overall_results' + '/' + model_name):
        os.mkdir(output_directory + 'overall_results' + '/' + model_name)
    if not path.exists(output_directory + 'overall_results' + '/' + model_name + '/' + prediction_type + '_' + dimension + '_' + PH):
        os.mkdir(output_directory + 'overall_results' + '/' + model_name + '/' + prediction_type + '_' + dimension + '_' + PH)

    return output_directory + 'overall_results' + '/' + model_name + '/' + prediction_type + '_' + dimension + '_' + PH
    
  
def main(model_name):
    overall_results = pd.DataFrame() #saving subject ID, test and validation RMSE for each iteration, overall mean RMSE and MAE 
    results_directory = make_directories()
    no_iterations = range(5)

    if normalize_data:
        substring = 'normalized_'+PH+'min'
    else:
        substring = PH+'min'

    if dataset == 'oaps':
        # Removing the filtered imputed from the path because we don't have that
        # print('Getting data from ', data_directory + seg + '\n')
        # unpickled_train_data = unpickle_data(data_directory + seg + 'windowed_train_' + substring + '.pickle') #e.g. windowed_train_normalized_60min.pickle
        # unpickled_test_data = unpickle_data(data_directory + seg + 'windowed_test_' + substring + '.pickle') 
        print('Getting data from ', data_directory + '\n')
        unpickled_train_data = unpickle_data(data_directory + 'windowed_train_' + substring + '.pickle') #e.g. windowed_train_normalized_60min.pickle
        unpickled_test_data = unpickle_data(data_directory + 'windowed_test_' + substring + '.pickle') 
    elif dataset == 'ohio':
        unpickled_train_data = unpickle_data(data_directory + 'OhioT1DM-training/imputed/'+'windowed_' + substring + '.pickle') #e.g. windowed_normalized_60min.pickle
        unpickled_test_data = unpickle_data(data_directory + 'OhioT1DM-testing/imputed/'+'windowed_' + substring + '.pickle')

    for i in no_iterations:
        print('Iteration #: ',i)
        results_df, model, model_train_history,model_val_history, all_subjs_predicted_values = CGM_Prediction(model_name,unpickled_train_data,unpickled_test_data,i )
        if i == no_iterations[0]:
            overall_results = pd.concat([results_df, overall_results], axis=1)
        else:
            overall_results = results_df.merge(overall_results, on='Subject', how='inner', suffixes=('_1', '_2'))
        if not path.exists(results_directory  + '/' + seg):
            os.mkdir(results_directory + '/' + seg)
        if not path.exists(results_directory + '/' + seg + str(i)):
            os.mkdir(results_directory + '/' + seg + str(i))
    
        filename = results_directory  + '/' + seg + str(i) + '/' + prediction_type + '_' + dimension + '_' + substring
        
        if save_results:
            with open(results_directory  + '/' + seg + str(i) + '/' + 'predicted_values.pickle', 'wb') as f:
                pickle.dump(all_subjs_predicted_values , f, pickle.HIGHEST_PROTOCOL)
            overall_results.to_csv(filename + '.csv')
            if model_type == 'deep':
                model.save(filename + '.h5')
            else:
                dump(model, filename+'.joblib') 
            if model_type == 'deep':
                model_history = {}
                model_history['train'] = model_train_history
                model_history['val'] = model_val_history
                with open(filename+'_history.pickle', 'wb') as f:
                    pickle.dump(model_history , f, pickle.HIGHEST_PROTOCOL)
    
    if save_results:
        exclude_keys = list()
        for key in overall_results.keys():
            if 'testRMSE' not in key:
                exclude_keys.append(key)

        test_rmse_df = overall_results.drop(exclude_keys, axis=1)
        overall_results['Mean Test RMSE']= test_rmse_df.mean(axis=1) 
        overall_results['STD Test RMSE']= test_rmse_df.std(axis=1) 

        exclude_keys = list()
        for key in overall_results.keys():
            if 'valRMSE' not in key:
                exclude_keys.append(key)

        val_rmse_df = overall_results.drop(exclude_keys, axis=1)
        overall_results['Mean Val RMSE']= val_rmse_df.mean(axis=1) 
        overall_results['STD Val RMSE']= val_rmse_df.std(axis=1) 

        filename = results_directory + '/' + seg + prediction_type + '_' + dimension + '_' + substring
        overall_results.to_csv(filename+'.csv')


if __name__ == "__main__":
    if len(sys.argv) > 4:
        root_directory = sys.argv[1]
        data_directory = sys.argv[2]
        output_directory = sys.argv[3]
        history_window = int(sys.argv[4]) #12
        prediction_window = int(sys.argv[5]) #30 or 60 minutes
        dimension = sys.argv[6] #univariate or multivariate
        prediction_type = sys.argv[7] #single or multi (single-step or multi-output)
        if sys.argv[8] == 'False':
            normalize_data = False
        else:
            normalize_data = True
        model_name = sys.argv[9]
        dataset = sys.argv[10]
        if sys.argv[11] == 'False':
            save_results = False
        else:
            save_results = True

        PH = str(prediction_window) #prediction horizon
        # if prediction_window == 30 or prediction_window == 60:
        prediction_window = prediction_window//5

        if (model_name == 'RNN') or (model_name == 'LSTM'):
            model_type = 'deep'
        elif model_name == 'REG' or model_name == 'TREE' or model_name == 'SVR' or model_name == 'ENSEMBLE':
            model_type = 'baseline'
        else:
            print('Model not found. Please choose model name from [RNN, LSTM, Reg, Tree, SVR, Ensemble]')
            exit(-1)
        
        # if dimension == 'univariate':
        #     n_features = 1
        # elif dimension == 'multivariate':
        n_features = 5 # Only working with multivariate data?


        print("Starting experiments for the following settings: prediction_window: "+PH+"; dimension: "+dimension+"; prediction_type: "+prediction_type+"; normalize_data: "+str(normalize_data)+"; model_name: "+model_name+" ;dataset: "+dataset+"; save_results: "+str(save_results)+'; ablation code: '+seg+'\n')
        main(model_name)
        print("Experiments completed for the following settings: prediction_window: "+PH+"; dimension: "+dimension+"; prediction_type: "+prediction_type+"; normalize_data: "+str(normalize_data)+"; model_name: "+model_name+" ;dataset: "+dataset+"; save_results: "+str(save_results)+'; ablation code: '+seg+'\n')
    else:
        print("Invalid input arguments")
        exit(-1)