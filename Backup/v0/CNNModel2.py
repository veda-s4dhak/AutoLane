# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 18:10:38 2018

@author: HP_OWNER
"""

""" Convolutional Neural Network.
Build and train a convolutional neural network with TensorFlow.
This example is using TensorFlow layers API, see 'convolutional_network_raw' 
example for a raw implementation with variables.
"""

import tensorflow as tf
import numpy as np

# Image dimension is 1080x840
# Frame size will be: 60x60
# Total: 18x14 = 252 frames 

# Output Format: [P1, bx, by, bh, bw, P2, bx, by, bh, bw]
# Dimension of output: 18x14x10 

# Create the neural network
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict['images']
        
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
        
        # Input is a 1080x840x3 size image
        x = tf.reshape(x, shape=[-1, 1080, 840, 3])

        # Convolution Layer with 15 filters and a kernel size of 5
        # Output: 1076x836x15
        conv1 = tf.layers.conv2d(x, 15, 5, activation=tf.keras.layers.LeakyReLU(0.01), kernel_initializer = initializer,
                                 bias_initializer = initializer)
        
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        # Output: 538x418x15
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 15 filters and a kernel size of 3
        # Output: 536x416x15
        conv2 = tf.layers.conv2d(conv1, 15, 3, activation=tf.keras.layers.LeakyReLU(0.01), kernel_initializer = initializer,
                                 bias_initializer = initializer)
        
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        # Output: 268x208x15
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
        
        
        # Output: 266x206x15
        conv3 = tf.layers.conv2d(conv2, 15, 3, activation=tf.keras.layers.LeakyReLU(0.01), kernel_initializer = initializer,
                                 bias_initializer = initializer)
        
        # Output: 133x103x15
        conv3 = tf.layers.max_pooling2d(conv3, 2, 2)
        
        # Fully connected layers using conv layers
        # Output: 18x14x1000
        # Filter size: 116x90
        conv4 = tf.layers.conv2d(conv3, 1000, (116,90), activation=None, kernel_initializer = initializer,
                                 bias_initializer = initializer)
        
        # Output: 18x14x1000
        conv5 = tf.layers.conv2d(conv4, 1000, 1, activation=None, kernel_initializer = initializer,
                                 bias_initializer = initializer)
        
        # Output: 18x14x10
        conv6 = tf.layers.conv2d(conv5, 10, 1, activation = tf.sigmoid, kernel_initializer = initializer,
                                 bias_initializer = initializer)
        
        # Applies dropout to output
        out = tf.layers.dropout(conv6, rate=dropout, training=is_training)
        
        # out = tf.reshape(conv6, [-1,1])

    return out