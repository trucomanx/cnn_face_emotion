#!/bin/bash

PyCommand='#!/usr/bin/python3

json_filename="training_data_results.json"

model_list=["efficientnet_b3",
            "inception_v3",
            "inception_resnet_v2",
            "resnet_v2_50",
            "mobilenet_v3"
            ];

info_list=[ "train_categorical_accuracy",
            "val_categorical_accuracy",
            "test_categorical_accuracy",
            "train_loss",
            "val_loss",
            "test_loss"
            ];

sep=",";

image_ext=".eps";
'

OutDir='/mnt/8811f502-ae19-4dd8-8371-f1915178f581/Fernando/OUTPUTS/DOCTORADO2/cnn_face_emotion_1'

  
DName='multiple_ber2024-face'  


################################################################################

mkdir -p $OutDir/$DName/'multiple_training_validation_holdout'
echo "$PyCommand" | cat - 'main.py' > temp && mv temp $OutDir/$DName/'multiple_training_validation_holdout/main.py'

################################################################################
export TF_USE_LEGACY_KERAS=1 
export CUDA_VISIBLE_DEVICES=0

ipynb-py-convert multiple_training_hodout.ipynb multiple_training_hodout.py

python3 multiple_training_hodout.py --model 'efficientnet_b3'     --epochs 100 --patience 20 --batch-size  16 --seed 42 --dataset-name $DName --output-dir $OutDir
python3 multiple_training_hodout.py --model 'inception_resnet_v2' --epochs 100 --patience 20 --batch-size  16 --seed 42 --dataset-name $DName --output-dir $OutDir
python3 multiple_training_hodout.py --model 'inception_v3'        --epochs 100 --patience 25 --batch-size  16 --seed 12 --dataset-name $DName --output-dir $OutDir
python3 multiple_training_hodout.py --model 'mobilenet_v3'        --epochs 100 --patience 20 --batch-size  16 --seed 42 --dataset-name $DName --output-dir $OutDir
python3 multiple_training_hodout.py --model 'resnet_v2_50'        --epochs 100 --patience 20 --batch-size  16 --seed 42 --dataset-name $DName --output-dir $OutDir

rm -f multiple_training_hodout.py

