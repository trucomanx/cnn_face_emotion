# Export `kfold_validation.ipynb` to `kfold_validation.py`
jupyter nbconvert --to python kfold_validation.ipynb

#python kfold_validation.py --model mobilenet_v3        --epochs 50  --batch-size 32
#python kfold_validation.py --model efficientnet_b3     --epochs 50  --batch-size 4
python kfold_validation.py --model inception_v3        --epochs 50 --batch-size 8
#python kfold_validation.py --model inception_resnet_v2 --epochs 50  --batch-size 32
#python kfold_validation.py --model resnet_v2_50        --epochs 50  --batch-size 32
#python kfold_validation.py --model custom1             --epochs 50  --batch-size 32
#python kfold_validation.py --model custom_dense1       --epochs 50  --batch-size 32
#python kfold_validation.py --model custom_residual1    --epochs 50  --batch-size 32
#python kfold_validation.py --model custom_inception    --epochs 50  --batch-size 32

rm -f kfold_validation.py