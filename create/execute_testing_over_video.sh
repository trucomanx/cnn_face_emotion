#!/bin/bash

InFile='/mnt/8811f502-ae19-4dd8-8371-f1915178f581/Fernando/DATASET/TESE/PATIENT-RECOGNITION/PATIENT-VIDEOS/dataset-toy/drhouse_mini_cut.mp4'

OutDir='/mnt/8811f502-ae19-4dd8-8371-f1915178f581/Fernando/OUTPUTS/DOCTORADO2/cnn_face_emotion_1'

DName='multiple_ber2024-face'  

WDir='/mnt/8811f502-ae19-4dd8-8371-f1915178f581/Fernando/OUTPUTS/DOCTORADO2/cnn_face_emotion'

################################################################################
export TF_USE_LEGACY_KERAS=1 
export CUDA_VISIBLE_DEVICES=-1

ipynb-py-convert testing_over_video.ipynb testing_over_video.py

python3 testing_over_video.py --model 'efficientnet_b3'     --weights-file $WDir/$DName/'multiple_training_validation_holdout'/'efficientnet_b3'/'model_efficientnet_b3.h5'         --dataset-name $DName --output-dir $OutDir --input-file $InFile
python3 testing_over_video.py --model 'inception_resnet_v2' --weights-file $WDir/$DName/'multiple_training_validation_holdout'/'inception_resnet_v2'/'model_inception_resnet_v2.h5' --dataset-name $DName --output-dir $OutDir --input-file $InFile
python3 testing_over_video.py --model 'inception_v3'        --weights-file $WDir/$DName/'multiple_training_validation_holdout'/'inception_v3'/'model_inception_v3.h5'               --dataset-name $DName --output-dir $OutDir --input-file $InFile
python3 testing_over_video.py --model 'mobilenet_v3'        --weights-file $WDir/$DName/'multiple_training_validation_holdout'/'mobilenet_v3'/'model_mobilenet_v3.h5'               --dataset-name $DName --output-dir $OutDir --input-file $InFile
python3 testing_over_video.py --model 'resnet_v2_50'        --weights-file $WDir/$DName/'multiple_training_validation_holdout'/'resnet_v2_50'/'model_resnet_v2_50.h5'               --dataset-name $DName --output-dir $OutDir --input-file $InFile

rm -f testing_over_video.py

