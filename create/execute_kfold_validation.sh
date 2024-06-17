# Export `kfold_validation.ipynb` to `kfold_validation.py`
jupyter nbconvert --to python kfold_validation.ipynb

DATASET='mcfer_v1.0' #'affectnet' #'fer2013'

python3 kfold_validation.py --model mobilenet_v3        --dataset $DATASET --epochs 50  --batch-size 32
python3 kfold_validation.py --model efficientnet_b3     --dataset $DATASET --epochs 50  --batch-size 16 
python3 kfold_validation.py --model inception_v3        --dataset $DATASET --epochs 50 --batch-size 32
python3 kfold_validation.py --model inception_resnet_v2 --dataset $DATASET --epochs 50  --batch-size 32
python3 kfold_validation.py --model resnet_v2_50        --dataset $DATASET --epochs 50  --batch-size 32
#python3 kfold_validation.py --model custom1             --dataset $DATASET --epochs 200 --batch-size 128 
#python3 kfold_validation.py --model custom_dense1       --dataset $DATASET --epochs 50  --batch-size 32
#python3 kfold_validation.py --model custom_residual1    --dataset $DATASET --epochs 50  --batch-size 32
#python3 kfold_validation.py --model custom_inception    --dataset $DATASET --epochs 50  --batch-size 32

rm -f kfold_validation.py
