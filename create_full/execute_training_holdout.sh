#!/bin/bash

PyCommand='#!/usr/bin/python3

json_filename="training_data_results.json"

model_list=["efficientnet_b3",
            "inception_resnet_v2",
            "inception_v3",
            "mobilenet_v3",
            "resnet_v2_50"
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
MachinePath='/media/maquina02/HD/Dados/Fernando'
#MachinePath='/media/fernando/B0EA304AEA300EDA/Dados/Fernando'
#MachinePath='/mnt/8811f502-ae19-4dd8-8371-f1915178f581/Fernando'

OutDir=$MachinePath/'OUTPUTS/DOCTORADO2/cnn_face_emotion_full1'

FineT='true'


#DName='ber2024-face'
DName='full2024-face'    



if [ "$DName" = "ber2024-face" ]; then
    InTrD=$MachinePath/'DATASET/TESE/BER/BER2024/BER2024-FACE'
    InTrF='train.csv'
    InTsD=$MachinePath/'DATASET/TESE/BER/BER2024/BER2024-FACE'
    InTsF='test.csv'
fi

if [ "$DName" = "full2024-face" ]; then
    InTrD=$MachinePath/'DATASET/TESE'
    InTrF='train_face.csv'
    InTsD=$MachinePath/'DATASET/TESE'
    InTsF='test_face.csv'
fi


################################################################################
if [ "$FineT" = "true" ]; then
    SubDir='training_validation_holdout_fine_tuning'
else
    SubDir='training_validation_holdout'
fi

mkdir -p $OutDir/$DName/$SubDir
echo "$PyCommand" | cat - 'main.py' > temp && mv temp $OutDir/$DName/$SubDir/'main.py'

################################################################################
export TF_USE_LEGACY_KERAS=1 
export CUDA_VISIBLE_DEVICES=0

ipynb-py-convert training_holdout.ipynb training_holdout.py

#python3 training_holdout.py --model 'efficientnet_b3'     --fine-tuning $FineT --epochs 200 --patience 40 --seed 0 --batch-size 16 --dataset-train-dir $InTrD --dataset-train-file $InTrF --dataset-test-dir $InTsD --dataset-test-file $InTsF --dataset-name $DName --output-dir $OutDir
python3 training_holdout.py --model 'inception_resnet_v2' --fine-tuning $FineT --epochs 200 --patience 40 --seed 0 --batch-size 16 --dataset-train-dir $InTrD --dataset-train-file $InTrF --dataset-test-dir $InTsD --dataset-test-file $InTsF --dataset-name $DName --output-dir $OutDir
python3 training_holdout.py --model 'inception_v3'        --fine-tuning $FineT --epochs 200 --patience 40 --seed 0 --batch-size 16 --dataset-train-dir $InTrD --dataset-train-file $InTrF --dataset-test-dir $InTsD --dataset-test-file $InTsF --dataset-name $DName --output-dir $OutDir
#python3 training_holdout.py --model 'mobilenet_v3'        --fine-tuning $FineT --epochs 200 --patience 40 --seed 0 --batch-size 16 --dataset-train-dir $InTrD --dataset-train-file $InTrF --dataset-test-dir $InTsD --dataset-test-file $InTsF --dataset-name $DName --output-dir $OutDir
#python3 training_holdout.py --model 'resnet_v2_50'        --fine-tuning $FineT --epochs 200 --patience 40 --seed 0 --batch-size 16 --dataset-train-dir $InTrD --dataset-train-file $InTrF --dataset-test-dir $InTsD --dataset-test-file $InTsF --dataset-name $DName --output-dir $OutDir

rm -f training_holdout.py

