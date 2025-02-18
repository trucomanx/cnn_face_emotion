#!/bin/bash

PyCommand='#!/usr/bin/python3

json_filename="testing_data_results.json"

model_list=["efficientnet_b3",
            "inception_resnet_v2",
            "inception_v3",
            "mobilenet_v3",
            "resnet_v2_50"
            ];

info_list=[ "block_delayms",
            "categorical_accuracy",
            "loss"];

sep=",";

image_ext=".eps";
'

#BaseDir='/media/fernando/Expansion'
BaseDir='/mnt/8811f502-ae19-4dd8-8371-f1915178f581/Fernando'

OutDir=$BaseDir'/OUTPUTS/DOCTORADO2/cnn_face_emotion4_1_10times'

SubTitle='fine_tuning'

DName='multiple_ber2024-face'
#DName='ber2024-face'  


if [ "$DName" = "ber2024-face" ] || [ "$DName" = "multiple_ber2024-face" ]; then
    InTsD=$BaseDir'/DATASET/TESE/BER/BER2024/BER2024-FACE'
    InTsF='test.csv'
    ModD=$BaseDir'/OUTPUTS/DOCTORADO2/FACE/cnn_face_emotion/multiple_ber2024-face/multiple_training_validation_holdout_step2'
fi

################################################################################

if [ "$SubTitle" = "" ]; then
    BaseDir='test_holdout'
else
    BaseDir='test_holdout_'$SubTitle
fi

mkdir -p $OutDir/$DName/$BaseDir
echo "$PyCommand" | cat - 'main.py' > temp && mv temp $OutDir/$DName/$BaseDir/'main.py'

################################################################################
export TF_USE_LEGACY_KERAS=1 

ipynb-py-convert testing_holdout.ipynb testing_holdout.py

python3 testing_holdout.py --model 'efficientnet_b3'     --model-file $ModD/'efficientnet_b3/model_efficientnet_b3.h5'         --times 10 --batch-size  16 --dataset-dir $InTsD --dataset-file $InTsF --dataset-name $DName --output-dir $OutDir --output-subtitle $SubTitle
python3 testing_holdout.py --model 'inception_resnet_v2' --model-file $ModD/'inception_resnet_v2/model_inception_resnet_v2.h5' --times 10 --batch-size  16 --dataset-dir $InTsD --dataset-file $InTsF --dataset-name $DName --output-dir $OutDir --output-subtitle $SubTitle
python3 testing_holdout.py --model 'inception_v3'        --model-file $ModD/'inception_v3/model_inception_v3.h5'               --times 10 --batch-size  16 --dataset-dir $InTsD --dataset-file $InTsF --dataset-name $DName --output-dir $OutDir --output-subtitle $SubTitle
python3 testing_holdout.py --model 'mobilenet_v3'        --model-file $ModD/'mobilenet_v3/model_mobilenet_v3.h5'               --times 10 --batch-size  16 --dataset-dir $InTsD --dataset-file $InTsF --dataset-name $DName --output-dir $OutDir --output-subtitle $SubTitle
python3 testing_holdout.py --model 'resnet_v2_50'        --model-file $ModD/'resnet_v2_50/model_resnet_v2_50.h5'               --times 10 --batch-size  16 --dataset-dir $InTsD --dataset-file $InTsF --dataset-name $DName --output-dir $OutDir --output-subtitle $SubTitle

rm -f testing_holdout.py

