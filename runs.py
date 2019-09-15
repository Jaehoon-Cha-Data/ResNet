# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 22:32:07 2019

@author: jaehooncha

@email: chajaehoon79@gmail.com

resnet runs
"""
import tensorflow as tf
from cifar10 import Cifar10
from collections import OrderedDict
import argparse
import os
import numpy as np
from models import ResNet
np.random.seed(0)
tf.set_random_seed(0)


def parse_args():
    parser = argparse.ArgumentParser()
    # optim config
    parser.add_argument('--model_name', type=str, default = 'ResNet')
    parser.add_argument('--datasets', type = str, default = 'CIFAR10')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--lr_sets', type = list, default = [0.01,
                                                             0.005,
                                                             0.001,
                                                                    5*1e-4,
                                                                    1e-4,
                                                                    5*1e-5])    
    parser.add_argument('--bottleneck', type = bool, default = False)
    parser.add_argument('--num_classes', type = int, default = 10)
    parser.add_argument('--num_outputs', type = int, default = 16)
    parser.add_argument('--kernel_size', type = int, default = 3)
    parser.add_argument('--conv_stride', type = int, default = 1)
    parser.add_argument('--block_sizes', type = list, default = [3,4,6,3])
    parser.add_argument('--block_strides', type = list, default = [1,2,2,2])

    args = parser.parse_args()
    
    config = OrderedDict([
        ('model_name', args.model_name),
        ('datasets', args.datasets),
        ('epochs', args.epochs),
        ('batch_size', args.batch_size),
        ('base_lr', args.base_lr),
        ('lr_sets', args.lr_sets),
        ('bottleneck', args.bottleneck),
        ('num_classes', args.num_classes),
        ('num_outputs', args.num_outputs),
        ('kernel_size', args.kernel_size),
        ('conv_stride', args.conv_stride),
        ('block_sizes', args.block_sizes),
        ('block_strides', args.block_strides)])

    return config


config = parse_args()


### call data ###
cifar10 = Cifar10()
n_samples = cifar10.num_examples 
n_test_samples = cifar10.num_test_examples

### call models ###
model = ResNet(config['bottleneck'], config['num_classes'], config['num_outputs'], 
               config['kernel_size'], config['conv_stride'], config['block_sizes'], config['block_strides'])   


### make folder ###
mother_folder = config['model_name']
try:
    os.mkdir(mother_folder)
except OSError:
    pass    


### outputs ###
pred, loss = model.Forward()

#lr = config['base_lr']

lr_sets = config['lr_sets']

for lr in lr_sets:
    folder_name = os.path.join(mother_folder, config['model_name']+'_'+config['datasets']+f'_{lr}')
    
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
    
    prediction = tf.argmax(pred, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(model.y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')
    tf.summary.scalar('accuracy', accuracy)
    
    summ = tf.summary.merge_all()
        
    writer_save_name = os.path.join('log', folder_name+'aug')
    writer = tf.summary.FileWriter(writer_save_name)
    
    
    train_loss_set = []
    train_acc_set = []
    test_loss_set = []
    test_acc_set = []    
    with tf.Session() as sess:    
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
    
        model_save_name = os.path.join(folder_name, config['model_name']+f'_{lr}'+'.ckpt')
        
        try:
            os.mkdir(folder_name)
        except OSError:
            pass    
        
        iteration = 0
        iter_per_epoch = n_samples/config['batch_size'] 
        iter_per_test_epoch = n_test_samples/config['batch_size'] 
        for epoch in range(config['epochs']):
            epoch_loss = 0.
            epoch_acc = 0.
            for iter_in_epoch in range(int(iter_per_epoch)):
                epoch_x, epoch_y = cifar10.next_train_batch(config['batch_size'])
                _, c, acc, s = sess.run([optimizer, loss, accuracy, summ], 
                                feed_dict = {model.x: epoch_x, model.y: epoch_y, model.training:True})
                writer.add_summary(s, global_step=iteration)
                epoch_loss += c
                epoch_acc += acc
                iteration+=1
                if iter_in_epoch%100 == 0:
                    print('Epoch ', epoch, '{:.2f}%'.format(100*(iter_in_epoch+1)/int(iter_per_epoch)),
                          'completed out of ', config['epochs'], 'loss: ', epoch_loss/(iter_in_epoch+1),
                          'acc: ', '{:.2f}%'.format(epoch_acc*100/(iter_in_epoch+1)))
            print('######################')        
            print('TRAIN')        
            print('Epoch ', epoch, '{:.2f}%'.format(100*(iter_in_epoch+1)/int(iter_per_epoch)),
                  'completed out of ', config['epochs'], 'loss: ', epoch_loss/int(iter_per_epoch),
                  'acc: ', '{:.2f}%'.format(epoch_acc*100/int(iter_per_epoch)))

            train_loss_set.append(epoch_loss/int(iter_per_epoch))
            train_acc_set.append(epoch_acc*100/int(iter_per_epoch))            
            test_loss = 0.
            test_acc = 0.
            for iter_in_epoch in range(int(iter_per_test_epoch)):            
                epoch_x, epoch_y = cifar10.next_test_batch(config['batch_size'])  
                c, acc = sess.run([loss, accuracy], 
                                  feed_dict = {model.x: epoch_x, model.y: epoch_y, model.training:False})
                test_loss += c
                test_acc += acc
            print('TEST')        
            print('Epoch ', epoch,  'loss: ', test_loss/int(iter_per_test_epoch), 
                  'acc: ', '{:.2f}%'.format(test_acc*100/int(iter_per_test_epoch)))
            print('###################### \n')     
            test_loss_set.append(test_loss/int(iter_per_test_epoch))
            test_acc_set.append(test_acc*100/int(iter_per_test_epoch))
    
            if epoch % 50 == 0:
                saver.save(sess, model_save_name, global_step=epoch)    
    
    path_name = os.path.join(folder_name, config['model_name']+f'_{lr}')
         

    with open(path_name+"_train_loss.txt", "wb") as f:    #Pickling
        pickle.dump(train_loss_set, f) 
    with open(path_name+"_test_loss.txt", "wb") as f:    #Pickling
        pickle.dump(test_loss_set, f) 
    with open(path_name+"_train_acc.txt", "wb") as f:    #Pickling
        pickle.dump(train_acc_set, f) 
    with open(path_name+"_test_acc.txt", "wb") as f:    #Pickling
        pickle.dump(test_acc_set, f) 
            
              
