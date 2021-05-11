# L-SSPQ
source code for the paper "Optimizing Lightweight Spatial Shift Point-Wise Quantization Model (SSPQ-v2)"

# introduction
This repository is related with the paper "Optimizing Lightweight Spatial Shift Point-Wise Quantization"

The main idea is
 - based on SSPQ and refer to the github(https://github.com/Eunhui-Kim/SSPQ) and paper(https://ieeexplore.ieee.org/document/9260175).
 - Lightweight SSPQ by
   Reducing learning parameters of fully-connected layer (linear- layer) and additionally applying quantization to linear layer.
 - Optimizing SSPQ learning by
   Efficient two-step learning for quantized neural network according to learning-weight states.
 - in detail, refer to the paper - ealry access (https://ieeexplore.ieee.org/document/9423989)
 
 
 
# our experimental Results for SSPQ 50 
L-SSPQ50 model achieves the best performance in the metric of information density.
<br>
<img width=50% height=50% src="https://github.com/Eunhui-Kim/L-SSPQ/blob/main/Information%20Density%200114.png" ></img>
<img width=50% height=50% src="https://github.com/Eunhui-Kim/L-SSPQ/blob/main/MB_accuracy%200114.png" ></img>
<br>
 
# pre-requisite
Note that this code is tested only in the environment decribed below. Mismatched versions does not guarantee correct execution.

 - Ubuntu kernel ver. 4.15.0-117-generic #118~16.04.1
 - Cuda 10.0
 - cudnn 7.6.5
 - Tensorflow 1.15.3
 - Tensorpack 
 - g++ 7.5.0
 - python 3.7
 - install https://github.com/Eunhui-Kim/custom-op-for-shiftNet
   
   After the test works fine, at first 
   
 - put the shift2d.py in the tensorpack models path.
   in my case, the tensorpack models is
   $HOME/.local/lib/python3.7/site-packages/tensorpack/models/

# Testing
  sspq50 model for imagenet dataset, 
  1) put 'imagenet-SSQ_S1.py, imagenet-SSQ_S2.py, resnet_model3.py, resnet_model4.py, with imagenet_utils.py and dorefa.py of tensorpack file together in the tensorpack example path
  >> run TS.sh
  
  sspq34 model for cifar100 dataset, 
  >> run SSQ_cifar100_TS.sh
  
