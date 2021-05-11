#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: cifar10-ResIBShift20
# Author : Eunhui Kim <kim.eunhui@gmail.com>

import argparse
import numpy as np
import os
import tensorflow as tf
import time

from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.tfutils.varreplace import remap_variables

from dorefa import get_dorefa
from tensorpack.models import shift2d 
#from tensorflow_active_shift.python.ops import active_shift2d_ops
#from resnet_model import get_bn

BATCH_SIZE = 256 
CLASS_NUM = 100

LR_SCHEDULE = [(0, 0.1), (100, 0.01), (150, 0.001)]
#LR_SCHEDULE = [(0, 0.001), (30, 0.0001), (60, 0.00001), (90, 0.000001)]
WEIGHT_DECAY = 1e-4

#FILTER_SIZES = [64, 128, 256]
#FILTER_SIZES = [64, 128, 256, 512]
MODULE_SIZES = [3, 4, 6, 3]
FILTER_SIZES = [16, 32, 64, 128]
#MODULE_SIZES = [2, 2, 3, 2]
#MODULE_SIZES = [3, 4, 2]

BITW = 8
BITA = 8
BITG = 32 

def get_bn(zero_init=False):
	 """
	     Zero init gamma is good for resnet. See https://arxiv.org/abs/1706.02677.
	 """
	 if zero_init:
		  return lambda x, name=None: BatchNorm('bn', x,  gamma_initializer=tf.zeros_initializer())
	 else:
		  return lambda x, name=None: BatchNorm('bn', x)

def resnet_shortcut(l, n_out, stride, activation=tf.identity):
    data_format = get_arg_scope()['Conv2D']['data_format']
    n_in = l.get_shape().as_list()[1 if data_format in ['NCHW', 'channels_first'] else 3]
    if n_in != n_out:   # change dimension when channel is not the same
        return Conv2D('convshortcut', l, n_out, 1, strides=stride, activation=activation)
    else:
        return l

class SSPQ_Cifar(ModelDesc):
    def inputs(self):
        return [tf.TensorSpec([None, 32, 32, 3], tf.float32, 'input'),
                tf.TensorSpec([None, CLASS_NUM], tf.float32, 'label')]

    def build_graph(self, image, label):
        assert tf.test.is_gpu_available()

        MEAN_IMAGE = tf.constant([0.4914, 0.4822, 0.4465], dtype=tf.float32)
        STD_IMAGE = tf.constant([0.2023, 0.1994, 0.2010], dtype=tf.float32)
        image = ((image / 255.0) - MEAN_IMAGE) / STD_IMAGE
        image = tf.transpose(image, [0, 3, 1, 2])

        fw, fa, fg = get_dorefa(BITW, BITA, BITG)

        def new_get_variable(v):
            name = v.op.name
            #skip binarize the first and last layer
            if not name.endswith('W') or 'conv1' in name or 'fct' in name:
               return v
            else:
               logger.info("Binarized weight {}".format(v.op.name))
               return fw(v)

        def new_w(v):
            return fw(v)

        def nonlin(x):
            return tf.clip_by_value(x, 0.0, 1.0)

        def activate(x):
            return fa(nonlin(x))

        def apply_preactivation2(l, preact):
            if preact == 'bnrelu':
                shortcut = l  # preserve identity mapping
                l = BatchNorm('preact', l)
                l = activate(l)
            else:
                shortcut = l
            return l, shortcut

        def resblock(x, channel, stride, preact):
            def get_stem_shift(x):
                net = Conv2D('conv1x1a', x, channel, kernel_size=1, strides=1, use_bias=False)
                net = BatchNorm('stembn', net)
                net = activate(net)
                net = Shift2D('shift_q', net, shift_size=channel, stride=stride)
                net = Conv2D('conv1x1b', net, channel, kernel_size=1, strides=1)
                net = BatchNorm('stembn2', net)
                return net

            x, shortcut = apply_preactivation2(x, preact)
            stem = get_stem_shift(x)
            return stem + resnet_shortcut(shortcut, channel, stride)

        def group(x, name, channel, i, blocks_in_module):
            for j in range(blocks_in_module):
               stride = 2 if i > 0 and j == 0 else 1 
               with tf.variable_scope(name + 'blk{}'.format(j+1)):
                 x = resblock(x, channel, stride, 'no_preact' if i==0 else 'bnrelu')

            # end of each group need an extra activation
            with tf.variable_scope(name):
                x = BatchNorm('bnlast', x)
                x = activate(x)
            return x

        def beforeAP(x, name, channel, i, blocks_in_module):
            stride = 1
            with tf.variable_scope(name + 'blk{}'.format(1)):
                x = Conv2D('conv1x1a', x, channel, kernel_size=1, strides=1)
                x = BatchNorm('stembn', x)
                x = activate(x)
            return x

        def afterAP(x, name, channel, i, blocks_in_module):
            stride = 1
            with tf.variable_scope(name + 'blk{}'.format(1)):
                #x = Conv2D('c1x1a', x, channel, kernel_size=1, strides=1)
                #x = BatchNorm('stembn', x)
                #x = Conv1D('')
                x = activate(x)
            return x

        pytorch_default_init = tf.variance_scaling_initializer(scale=1.0 / 3, mode='fan_in', distribution='uniform')
        with remap_variables(new_get_variable), \
             argscope(BatchNorm, decay=0.9, epsilon=1e-4), \
             argscope([Conv2D, BatchNorm, GlobalAvgPooling], data_format='channels_first'), \
             argscope(Conv2D, kernel_initializer=pytorch_default_init), \
             argscope(Shift2D, n_shift=2, padding='SAME'):
             image = Conv2D('conv1', image, 16, kernel_size=3, strides=1, use_bias=False)
             logits = (LinearWrap(image)
                     #.Conv2D('conv1', 16, 3, strides=1, use_bias=False)
                     .apply(group, 'conv2', 16, 0, 3)
                     .apply(group, 'conv3', 32, 1, 4)
                     .apply(group, 'conv4', 64, 2, 6)
                     #.apply(group, 'conv5', 128, 3, 3)
                     .apply(group, 'conv5', 128, 3, 2)
                     .apply(beforeAP, 'conv6', 128, 4, 1)
                     .GlobalAvgPooling('gap')
                     .apply(afterAP, 'conv7', 128, 5, 1)
                     .FullyConnected('fct', CLASS_NUM, kernel_initializer=tf.random_normal_initializer(stddev=1e-3))())   

        ce_cost = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits)
        ce_cost = tf.reduce_mean(ce_cost, name='cross_entropy_loss')

        single_label = tf.cast(tf.argmax(label, axis=1), tf.int32)
        wrong = tf.cast(tf.logical_not(tf.nn.in_top_k(logits, single_label, 1)), tf.float32, name='wrong_vector')
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'), ce_cost)
        add_param_summary(('.*/W', ['histogram']))
        add_param_summary(('.*/SP', ['histogram']))

        # weight decay on all W matrixes. including convolutional layers
        wd_cost = tf.multiply(WEIGHT_DECAY, regularize_cost('.*', tf.nn.l2_loss), name='wd_cost')

        return tf.add_n([ce_cost, wd_cost], name='cost')

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        add_moving_summary(tf.reduce_mean(lr, name='learning_rate'))
        return opt


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = dataset.Cifar100(train_or_test)
    pp_mean = ds.get_per_pixel_mean(('train',))
    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    
    def f(dp):
        images, labels = dp 
        one_hot_labels = np.eye(CLASS_NUM)[labels]
        return [images, one_hot_labels]

    ds = MapData(ds, f)
    return ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--log', default='train_log/ResIBShift34-dorefa')
    parser.add_argument('--eta_min', help='learning rate eta', default=1e-3)
    parser.add_argument('--max_epoch', default=200)
    parser.add_argument('--base_lr', default=0.1)
    parser.add_argument('--wup_length', default=5)
    parser.add_argument('--step_size', default=5)
    #parser.add_argument('--shuffle', default=False)
    args = parser.parse_args()

    start_t = time.time()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    log_folder = args.log
    logger.set_logger_dir(os.path.join(log_folder))

    dataset_train = get_data('train')
    dataset_test = get_data('test')

    max_epoch = int(args.max_epoch)
    eta_min = float(args.eta_min)
    base_lr = float(args.base_lr)
    warmup_len = int(args.wup_length)
    step_size = int(args.step_size)
    config = TrainConfig(
        model=SSPQ_Cifar(),
        #data=QueueInput(dataset_train),
        dataflow = dataset_train, 
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            [ScalarStats('cost'), ClassificationError('wrong_vector')]),
            #ScheduledHyperParamSetter('learning_rate', LR_SCHEDULE)
            HyperParamSetterWithFunc('learning_rate',
                                     lambda e, x: x * (eta_min + (1 - eta_min) * (
                                             1 + np.cos(np.pi * (e - warmup_len) / (max_epoch - warmup_len))) / 2.0)
                                     if e > warmup_len and x >= eta_min and e % step_size == 0 else (
                                         eta_min if x < eta_min else x))
        ],
        max_epoch=200,
        steps_per_epoch=len(dataset_train),
        session_init=SmartInit(args.load)
    )
    #launch_train_with_config(config, SimpleTrainer())
    num_gpu = max(get_num_gpu(), 1)
    launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(num_gpu))
    tt = time.time() - start_t
    print('total_train time: {} min'.format(tt/60.0))
