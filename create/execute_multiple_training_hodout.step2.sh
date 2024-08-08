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
SubDir='multiple_training_validation_holdout_step2'

WDir='/mnt/8811f502-ae19-4dd8-8371-f1915178f581/Fernando/OUTPUTS/DOCTORADO2/cnn_face_emotion/multiple_ber2024-face/multiple_training_validation_holdout'

################################################################################

mkdir -p $OutDir/$DName/$SubDir
echo "$PyCommand" | cat - 'main.py' > temp && mv temp $OutDir/$DName/$SubDir/'main.py'

################################################################################
export TF_USE_LEGACY_KERAS=1 
export CUDA_VISIBLE_DEVICES=0

ipynb-py-convert multiple_training_hodout.ipynb multiple_training_hodout.py

for ModelType in 'mobilenet_v3' 'resnet_v2_50'; do
    echo " "
    echo " "
    python3 multiple_training_hodout.py --model $ModelType \
                                        --epochs 100 --patience 20 \
                                        --batch-size 16 \
                                        --seed 42 \
                                        --weights-path $WDir/$ModelType/'model_'$ModelType'.h5' \
                                        --dataset-name $DName \
                                        --subdir $SubDir \
                                        --output-dir $OutDir
done

rm -f multiple_training_hodout.py

