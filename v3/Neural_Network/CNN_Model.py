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
import copy as cp


# ================================ LOCAL IMPORTS ================================ #

import sys

sys.path.insert(0, r'C:\\Users\\Veda Sadhak\\Desktop\\LOL-Autolane\\v3\\Data_Set')

import data_labeller as dlb
import data_set as ds

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
                is_realTime
                ):
        self.sess = sess
        self.is_train = is_train
        self.is_realTime = is_realTime
        self.test_dir = test_dir

        # Specifies image parameters
        self.imageHeight = 300
        self.imageWidth = 400
        self.numPartsX = 20 #TODO check how big bounding boxes are in general
        self.numPartsY = 20

        self.images = tf.placeholder(tf.float32,[None,self.imageHeight,self.imageWidth,3],name='images')
        self.labels = tf.placeholder(tf.float32,[None,self.numPartsY,self.numPartsX,1],name='labels')

        # Specifies training parameters
        self.epoch = 400
        self.batch_size = 3
        self.learning_rate = 1e-5
        self.drop_prob = 0.25

        # Creates the model
        self.pred = self.conv_net()
        # self.loss = -tf.reduce_mean(self.labels*tf.log(self.pred+1e-10) + (1-self.labels)*tf.log(1-(self.pred+1e-10)))/self.batch_size
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
        self.saver = tf.train.Saver()
        
        images =  ds.load_images()
        print(len(images))
        label_matrix = dlb.load_matrix()
        self.shuffled_images, self.shuffled_label_matrix = ds.prepareDataSet(images, label_matrix, 25,25, 0)
        print(len(self.shuffled_images[0]))
        print(len(self.shuffled_images[1]))
        self.shuffled_label_matrix[0] = cp.deepcopy(self.shuffled_label_matrix[0].reshape((-1, self.numPartsY, self.numPartsX, 1)))
        self.shuffled_label_matrix[1] = cp.deepcopy(self.shuffled_label_matrix[1].reshape((-1, self.numPartsY, self.numPartsX, 1)))

    # Create the network architecture (we feed this into the train function)
   def conv_net(self):
    
        # TF Estimator input is a dict, in case of multiple inputs
        x = self.images
    
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
    
        # Input is a 300x400x3 size image
        x = tf.reshape(x, shape=[-1, self.imageHeight, self.imageWidth, 3])
    
        # Convolution Layer with 15 filters and a kernel size of 5
        # Output: 296x396x15
        conv1 = tf.layers.conv2d(x, 15, 5, activation=tf.keras.layers.LeakyReLU(0.01), kernel_initializer = initializer,
                                    bias_initializer = initializer)
    
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        # Output: 148x198x15
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
    
        # Convolution Layer with 15 filters and a kernel size of 3
        # Output: 146x196x15
        conv2 = tf.layers.conv2d(conv1, 15, 3, activation=tf.keras.layers.LeakyReLU(0.01), kernel_initializer = initializer,
                                    bias_initializer = initializer)
    
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        # Output: 73x98
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
    
    
        # Output: 41x41x15
        conv3 = tf.layers.conv2d(conv2, 15, (33, 58), activation=tf.keras.layers.LeakyReLU(0.01), kernel_initializer = initializer,
                                    bias_initializer = initializer)
    
        # Output: 20x20x15
        conv3 = tf.layers.max_pooling2d(conv3, 2, 2)
    
        # Fully connected layers using conv layers
        # Output: 20x20x1000
        conv4 = tf.layers.conv2d(conv3, 1000, 1, activation=None, kernel_initializer = initializer,
                                    bias_initializer = initializer)
    
        # Output: 20x20x1000
        conv5 = tf.layers.conv2d(conv4, 1000, 1, activation=None, kernel_initializer = initializer,
                                    bias_initializer = initializer)
    
        # Output: 20x20x1
        conv6 = tf.layers.conv2d(conv5, 1, 1, activation = tf.sigmoid, kernel_initializer = initializer,
                                    bias_initializer = initializer)
    
        # Applies dropout to output
        out = tf.layers.dropout(conv6, rate=self.drop_prob, training=self.is_train)
    
        # out = tf.reshape(conv6, [-1,1])
    
        return out
   def runRealTime(self, config):
       '''images =  ds.load_images()
       label_matrix = dlb.load_matrix()
       self.shuffled_images, self.shuffled_label_matrix = ds.prepareDataSet(images, label_matrix, 18, 18, 0)
        
       self.shuffled_label_matrix[0] = cp.deepcopy(self.shuffled_label_matrix[0].reshape((-1, self.numPartsY, self.numPartsX, 1)))
       self.shuffled_label_matrix[1] = cp.deepcopy(self.shuffled_label_matrix[1].reshape((-1, self.numPartsY, self.numPartsX, 1)))
       print(len(self.shuffled_images), len(self.shuffled_label_matrix))'''
       self.sess.run(tf.global_variables_initializer())
       
       counter = 0
       time_ = time.time()
    
       self.load(config.checkpoint_dir)
       
       print("Now Start Realtime Running...")
       
       
       
   def train(self, config):
    
        # NOTE : if train, the nx, ny are ingnored
    
        # data_dir = checkpoint_dir(config)
    
        images =  ds.load_images()
        label_matrix = dlb.load_matrix()
        shuffled_images, shuffled_label_matrix = ds.prepareDataSet(images, label_matrix, 18, 18, 0)
        
        shuffled_label_matrix[0] = cp.deepcopy(shuffled_label_matrix[0].reshape((-1, self.numPartsY, self.numPartsX, 1)))
        shuffled_label_matrix[1] = cp.deepcopy(shuffled_label_matrix[1].reshape((-1, self.numPartsY, self.numPartsX, 1)))
        
        print(len(shuffled_images), len(shuffled_label_matrix))
    
        # Stochastic gradient descent with the standard backpropagation
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
    
        counter = 0
        time_ = time.time()
    
        self.load(config.checkpoint_dir)
        # Train
        if config.is_train:
            print("Now Start Training...")
            for ep in range(self.epoch):
                # Run by batch images
                
                batch_idxs = np.shape(shuffled_images[0])[0]//self.batch_size # config.batch_size
                print(batch_idxs)
                for idx in range(0, batch_idxs):
                    batch_images = shuffled_images[0][idx * self.batch_size : (idx + 1) * self.batch_size]
                    print(np.shape(batch_images))
                    batch_labels = shuffled_label_matrix[0][idx * self.batch_size : (idx + 1) * self.batch_size]
                    print(np.shape(batch_labels))
                    counter += 1
                    _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})
    
                    if counter % 10 == 0:
                        print("Epoch: ", (ep+1), " Step: ", counter, " Time: ", (time.time()-time_), " Loss: ", err)
                        #print(label_[1] - self.pred.eval({self.images: input_})[1],'loss:]',err)
                    if counter % 500 == 0:
                        self.save(config.checkpoint_dir, counter)
        # Test
        else:
            print("Now Start Testing...")
            
            for i in range(18):
                print('Test Image: ', i)
                print('Result:')
                result = self.pred.eval({self.images: np.expand_dims(shuffled_images[1][i], 0)})
                print(result)
                labelMat = (result > 0.5).astype(np.float32)
                print('Threshold Filtered Result')
                print(labelMat)
                print('Error: ' ,np.linalg.norm((labelMat-shuffled_label_matrix[1][i]), ord=None))
                
                
    
    
    
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
   def load(self, checkpoint_dir):
        """
            To load the checkpoint use to test or pretrain
        """
        print("\nReading Checkpoints.....\n\n")
        model_dir = "%s" % ("cnn") # give the model name by label_size
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    
        # Check the checkpoint is exist
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = str(ckpt.model_checkpoint_path) # convert the unicode to string
            self.saver.restore(self.sess, os.path.join(os.getcwd(), ckpt_path))
            print("\n Checkpoint Loading Success! %s\n\n"% ckpt_path)
        else:
            print("\n! Checkpoint Loading Failed \n\n")

