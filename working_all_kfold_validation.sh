
# Export `kfold_validation.ipynb` to `kfold_validation.py`
jupyter nbconvert --to python kfold_validation.ipynb

python kfold_validation.py --model mobilenet_v3
python kfold_validation.py --model efficientnet_b3
python kfold_validation.py --model inception_v3
python kfold_validation.py --model inception_resnet_v2
python kfold_validation.py --model resnet_v2_50

rm -f kfold_validation.py