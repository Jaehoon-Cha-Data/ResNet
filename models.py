# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 23:41:44 2019

@author: jaehooncha

@email: chajaehoon79@gmail.com

resnet model
"""
import tensorflow as tf
from networks import Batch_norm, Building_block, Bottleneck_block, Block_layer, Augmentation

class ResNet(object):
    def __init__(self, bottleneck, num_classes, num_outputs,
               kernel_size, conv_stride, block_sizes, block_strides):

        self.bottleneck = bottleneck
        if bottleneck:
            self.block_fn = Bottleneck_block
        else:
            self.block_fn = Building_block

        self.num_classes = num_classes
        self.num_outputs = num_outputs                #the first n_kernel_output
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.block_sizes = block_sizes                #block_sizes = [4,4,4]
        self.block_strides = block_strides            #block_strides = [1, 2, 2]
        self.n_blocks = len(self.block_sizes)
        self.final_size = num_outputs * (2 ** (len(block_sizes)-1))

        self.x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name = "resnet_input")
        self.y = tf.placeholder(tf.float32, shape=(None, 10), name = "resnet_one_hot")  # onehot
        self.training = tf.placeholder(tf.bool,shape=(), name = "training")

        self.Make_blocks()

    def Make_blocks(self):
        self.blocks = []
        for i in range(self.n_blocks):
            name='block_layer{}'.format(i + 1)
            block_num_outputs = self.num_outputs * (2**i)
            with tf.name_scope(name):                
                block = Block_layer(n_out_filters=block_num_outputs, bottleneck=self.bottleneck,
                    block_fn=self.block_fn, blocks=self.block_sizes[i],
                    strides=self.block_strides[i], training=self.training)
            self.blocks.append(block)
                
    def Connect_blocks(self, x):
        for i in range(self.n_blocks):
            x = self.blocks[i].Forward(x)
        return x


    def Forward(self):
        self.x = Augmentation(self.x, self.training)
        self.conv1 = tf.layers.conv2d(
          inputs=self.x, filters=self.num_outputs, kernel_size=self.kernel_size,
          strides=self.conv_stride, padding = 'SAME', name = 'conv1')

        with tf.name_scope('res_con_output'):
            self.res_con = self.Connect_blocks(self.conv1)

        self.res_con = Batch_norm(self.res_con, self.training)
        self.res_con = tf.nn.relu(self.res_con)
    
        # The current top layer has shape
        # `batch_size x pool_size x pool_size x final_size`.
        with tf.name_scope('mean_pool_output'):
            self.pool = tf.reduce_mean(self.res_con, [1, 2], keepdims=True)
        
        self.reshape = tf.reshape(self.pool, [-1, self.final_size])

        with tf.name_scope('prediction'):
            self.pred = tf.layers.dense(inputs=self.reshape, units=self.num_classes, name = 'final_output')
   
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.pred,
                                                                 labels=tf.stop_gradient([self.y])))

        def Summaray():
            tf.summary.scalar('loss', self.loss)

        Summaray()

        return self.pred, self.loss
