#!/bin/bash

PyCommand='#!/usr/bin/python3

json_filename="fold_status.json",#"kfold_data_results.json"

model_list=["efficientnet_b3",
            "inception_resnet_v2",
            "inception_v3",
            "mobilenet_v3",
            "resnet_v2_50"
            ];

info_list=[ "mean_val_categorical_accuracy",
            "std_val_categorical_accuracy",
            "mean_val_loss",
            "mean_train_categorical_accuracy",
            "mean_train_loss"];

erro_bar=[("mean_val_categorical_accuracy","std_val_categorical_accuracy")];

sort_by="val_categorical_accuracy";

p_matrix="val_categorical_accuracy";

sep=",";

image_ext=".eps";
'

OutDir='/mnt/8811f502-ae19-4dd8-8371-f1915178f581/Fernando/OUTPUTS/DOCTORADO2/cnn_face_emotion_1'

#DName='affectnet' 
DName='ber2024-face'




if [ "$DName" = "fer2013" ]; then
    InTrD='/mnt/8811f502-ae19-4dd8-8371-f1915178f581/Fernando/DATASET/EXTERN/FACE/fer2013/archive/train'
    InTrF='training_labels.csv'
fi

if [ "$DName" = "affectnet" ]; then
    InTrD='/mnt/8811f502-ae19-4dd8-8371-f1915178f581/Fernando/DATASET/EXTERN/FACE/AffectNet-Sample/input/affectnetsample/train_class'
    InTrF='training_labels.csv'
fi

if [ "$DName" = "mcfer_v1.0" ]; then
    InTrD='/mnt/8811f502-ae19-4dd8-8371-f1915178f581/Fernando/DATASET/TESE/FACE-EMOTION/mcfer/archive/train'
    InTrF='training_labels.csv'
fi

if [ "$DName" = "ber2024-face" ]; then
    InTrD='/mnt/8811f502-ae19-4dd8-8371-f1915178f581/Fernando/DATASET/TESE/BER/BER2024/BER2024-FACE'
    InTrF='train.csv'
fi

################################################################################

mkdir -p $OutDir/$DName/cross-validation
echo "$PyCommand" | cat - 'main.py' > temp && mv temp $OutDir/$DName/cross-validation/'main.py'

################################################################################
export TF_USE_LEGACY_KERAS=1 
export CUDA_VISIBLE_DEVICES=0

ipynb-py-convert kfold_validation.ipynb kfold_validation.py

#python3 kfold_validation.py --model 'mobilenet_v3'        --epochs 100  --batch-size 16 --dataset-dir $InTrD --dataset-file $InTrF --dataset-name $DName --output-dir $OutDir
#python3 kfold_validation.py --model 'efficientnet_b3'     --epochs 100  --batch-size 16 --dataset-dir $InTrD --dataset-file $InTrF --dataset-name $DName --output-dir $OutDir
#python3 kfold_validation.py --model 'inception_v3'        --epochs 100  --batch-size 16 --dataset-dir $InTrD --dataset-file $InTrF --dataset-name $DName --output-dir $OutDir
python3 kfold_validation.py --model 'inception_resnet_v2' --epochs 100  --batch-size 16 --dataset-dir $InTrD --dataset-file $InTrF --dataset-name $DName --output-dir $OutDir
python3 kfold_validation.py --model 'resnet_v2_50'        --epochs 100  --batch-size 16 --dataset-dir $InTrD --dataset-file $InTrF --dataset-name $DName --output-dir $OutDir
##python3 kfold_validation.py --model 'custom1'             --epochs 200 --batch-size 64 --dataset-dir $InTrD --dataset-file $InTrF --dataset-name $DName --output-dir $OutDir 
##python3 kfold_validation.py --model 'custom_dense1'       --epochs 50  --batch-size 16 --dataset-dir $InTrD --dataset-file $InTrF --dataset-name $DName --output-dir $OutDir
##python3 kfold_validation.py --model 'custom_residual1'    --epochs 50  --batch-size 16 --dataset-dir $InTrD --dataset-file $InTrF --dataset-name $DName --output-dir $OutDir
##python3 kfold_validation.py --model 'custom_inception'    --epochs 50  --batch-size 16 --dataset-dir $InTrD --dataset-file $InTrF --dataset-name $DName --output-dir $OutDir

rm -f kfold_validation.py

