#!/bin/sh
dataset='oaps' #ohio or oaps
if [ "$dataset" = "ohio" ]; then
    root_directory="../../../../PHI/PHI_OHIO/" 
    data_directory=$root_directory"data/"
fi
if [ "$dataset" = "oaps" ]; then
    root_directory=$PWD #"../../../../PHI/PHI_OAPS/" 
    data_directory=$root_directory"/n=183_OpenAPSDataCommonsAugust2021/" #../../../data/PHI/PHI_OAPS/OpenAPS_data/n=88_OpenAPSDataAugust242018/

fi
filter_data=True
normalize_data=False
threshold=15 #remove CGM readings below 15 mg/dL
history_window=12 #no. of past values to use to make estimations of future values
prediction_window=60 #no. of future values to predict (6 denotes a prediction horizon of 5 * 6 = 30 min)
python3 $PWD/preprocess_$dataset.py $root_directory $data_directory $filter_data $normalize_data $threshold $history_window $prediction_window