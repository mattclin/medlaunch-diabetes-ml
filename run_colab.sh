#!/bin/sh
dataset='oaps' #ohio or oaps
if [ "$dataset" = "ohio" ]; then
    root_directory=$PWD #"../../../../PHI/PHI_OHIO/" 
    data_directory=$root_directory"data/csv_files/" 
fi
if [ "$dataset" = "oaps" ]; then
    root_directory="$PWD/"
    data_directory=$root_directory"n=183_OpenAPSDataCommonsAugust2021/" #../../../data/PHI/PHI_OAPS/OpenAPS_data/n=183_OpenAPSDataCommonsAugust2021/

fi
output_directory="OHIO_models/"
model_directory='OAPS_models/'
history_window=12
prediction_window=30
dimension=univariate
prediction_type=single #single-step or multi-output
normalize_data=False
model_name=LSTM
save_results=False
mimic_pipeline="student"
if [ "$dataset" = "ohio" ]; then
    python $PWD/main.py $root_directory $data_directory $output_directory $model_directory $history_window $prediction_window $dimension $prediction_type $normalize_data $model_name $dataset $save_results $mimic_pipeline
fi
if [ "$dataset" = "oaps" ]; then
    # python $PWD/testpickle.py
    python $PWD/train_oaps.py $root_directory $data_directory $model_directory $history_window $prediction_window $dimension $prediction_type $normalize_data $model_name $dataset $save_results 1
fi
