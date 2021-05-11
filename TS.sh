python imagenet-SSQ_S1.py --mode QresIBShift --data $Imagenet_Path --batch 256 --depth 50
python imagenet-SSQ_S2.py --mode QresIBShift --data $Imagenet_Path --load ./train_log/imagenet-QresIBShift-d50-batch256-ssplq-clr-0510-q48-TS1/checkpoint --batch 256 --depth 50 --log imagenet-QresIBShift-d50-batch256-ssplq-clr-0510-q48-TS2_1
