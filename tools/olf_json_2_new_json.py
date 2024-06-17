#!/usr/bin/python3

import os
import json
import numpy as np

json_status_path='fold_status.json';

if os.path.isfile(json_status_path):
    os.rename(json_status_path, json_status_path+'.old')
    # Read JSON file
    with open(json_status_path+'.old') as data_file:
        data_fold = json.load(data_file)
    
    
    data_fold['val_categorical_accuracy'] = data_fold.pop('VALIDATION_ACCURACY');
    data_fold['val_loss'] = data_fold.pop('VALIDATION_LOSS');

    data_fold['mean_val_categorical_accuracy'] = np.mean(data_fold['val_categorical_accuracy']);
    data_fold['std_val_categorical_accuracy']  = np.std(data_fold['val_categorical_accuracy']);

    data_fold['mean_val_loss'] = np.mean(data_fold['val_loss']);
    data_fold['std_val_loss']  = np.std(data_fold['val_loss']);
    
    with open(json_status_path, 'w') as f:
        json.dump(data_fold, f,indent=4);


