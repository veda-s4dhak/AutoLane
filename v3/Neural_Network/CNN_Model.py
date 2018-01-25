'''

TODO

-> Conv_Net: Change neural network into a class (below are the methods required for the class)
    -> __Init__(Xiao Lei and Anish): Initialize model parameters (data based on flags, learning rate,
                                               Tensorflow session, initializes placeholder,
                                               weights, biases, cost function, saver)
    -> Train (Anish): Runs the training on the model
    -> Model (Xiao Lei): Contains the neural network
    -> Load (Anish): Loads the model for forward propagation
    -> Save (Xiao Lei): Saves the model during training
-> Change resolution of neural network to accomodate new dataset

'''




# ================================ GLOBAL IMPORTS ================================ #

import tensorflow as tf
import numpy as np
import time
import os

# ================================ GENERAL NOTES ================================ #

# Image dimension is 1080x840
# Frame size will be: 60x60
# Total: 18x14 = 252 frames 

# Output Format: [P1, bx, by, bh, bw, P2, bx, by, bh, bw]
# Dimension of output: 18x14x10 

# ================================ CONV NET ================================ #
class CNN_Model(object):
    
    def __init__(self,
                 sess,
                 is_train,
                 test_dir,
                 ):
        self.sess = sess
        self.is_train = is_train
        self.test_dir = test_dir
        
        # Specifies image parameters
        self.imageHeight = 344
        self.imageWidth = 258
        self.numPartsX = 43
        self.numPartsY = 43
        
        # Specifies training parameters
        self.epoch = 400
        self.batch_size = 5
        self.learning_rate = 1e-5
        self.drop_prob = 0.25
    
    
    # Create the neural network
    def conv_net(self):
        
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict['images']
            
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
            
        # Input is a 1080x840x3 size image
        x = tf.reshape(x, shape=[-1, self.imageHeight, self.imageWidth, 3])

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
        out = tf.layers.dropout(conv6, rate=self.drop_prob, training=self.is_train)
            
        # out = tf.reshape(conv6, [-1,1])
    
        return out
    
    def checkpoint_dir(config):
        if config.is_train:
            return os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")
        else:
            return os.path.join('./{}'.format(config.checkpoint_dir), "test.h5")
        
    def save(self, checkpoint_dir, step):
        """
            To save the checkpoint use to test or pretrain
        """
        model_name = "CNN.model"
        model_dir = "%s" % ("cnn")
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
             os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
