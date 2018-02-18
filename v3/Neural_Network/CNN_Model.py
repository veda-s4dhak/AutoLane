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

sys.path.insert(0, r'C:\\Users\\HP_OWNER\\Desktop\\LOL-Autolane\\v3\\Data_Set')

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
        self.epoch = 2000
        self.batch_size = 10
        self.learning_rate = 1e-5
        self.drop_prob = 0.25
        self.numTrain = 40
        self.numValid = 10
        self.numTest = 0

        # Creates the model
        self.pred = self.conv_net()
        
        
        eps = 1e-4
        # self.loss = -tf.reduce_mean(self.labels*tf.log(self.pred+1e-10) + (1-self.labels)*tf.log(1-(self.pred+1e-10)))/self.batch_size
        # self.loss = tf.reduce_mean(tf.square(self.labels - self.pred)  
        '''self.comparison = tf.less( self.pred, tf.constant(eps) )    
        self.ca_op = self.pred.assign( tf.where(self.comparison, (eps)*tf.ones_like(self.pred), self.pred ))
        
        self.comparison1 = tf.greater( self.ca_op, tf.constant(1-eps ))    
        self.ca_op1 = self.pred.assign( tf.where(self.comparison1, 1-(eps)*tf.ones_like(self.ca_op), self.ca_op ))'''
        
        
        self.loss = -tf.reduce_sum(8*self.labels*tf.log(self.pred+eps) + (1-self.labels)*tf.log(1-(self.pred)+eps))
        self.saver = tf.train.Saver()
        
        print('Loading images...')
        self.imagesData =  ds.load_images()
        
        print('Loading label matrices...')
        self.label_matrix = dlb.load_matrix()
        
        print('Preparing data...')
        self.shuffled_images, self.shuffled_label_matrix = ds.prepareDataSet(self.imagesData, self.label_matrix, self.numTrain, self.numValid, self.numTest)
        
        self.shuffled_images[0] = np.array(self.shuffled_images[0]).astype(np.float32)
        self.shuffled_images[1] = np.array(self.shuffled_images[1]).astype(np.float32)
        self.shuffled_label_matrix[0] = (np.array(self.shuffled_label_matrix[0]).astype(np.float32)).reshape((-1, self.numPartsY, self.numPartsX, 1))
        self.shuffled_label_matrix[1] = (np.array(self.shuffled_label_matrix[1]).astype(np.float32)).reshape((-1, self.numPartsY, self.numPartsX, 1))
        
        print(np.shape(self.shuffled_images[0]))
        print(np.shape(self.shuffled_images[1]))
        print(np.shape(self.shuffled_label_matrix[0]))
        print(np.shape(self.shuffled_label_matrix[1]))
        print(self.shuffled_images[0])
        print(self.shuffled_label_matrix[0])
        
        
        print(len(self.shuffled_images), len(self.shuffled_label_matrix))
        
        #images =  ds.load_images()
        #print(len(images))
        #label_matrix = dlb.load_matrix()
        #self.shuffled_images, self.shuffled_label_matrix = ds.prepareDataSet(images, label_matrix, 40,10, 0)
        #print(len(self.shuffled_images[0]))
        #print(len(self.shuffled_images[1]))
        #self.shuffled_label_matrix[0] = cp.deepcopy(self.shuffled_label_matrix[0].reshape((-1, self.numPartsY, self.numPartsX, 1)))
        #self.shuffled_label_matrix[1] = cp.deepcopy(self.shuffled_label_matrix[1].reshape((-1, self.numPartsY, self.numPartsX, 1)))

    # Create the network architecture (we feed this into the train function)
   def conv_net(self):
    
        # TF Estimator input is a dict, in case of multiple inputs
        x = self.images
        
        kw = [3,2,3,2,3,1,3,2,3,1,3,2,3,1,3,1,3,2,3,1,1,1,1,1,1,1,1,1]
        kh = [3,2,3,2,3,1,3,2,3,1,3,2,3,1,3,1,3,2,3,1,1,1,1,1,1,1,1,1]
        
        
        kw[19] = 3
        kw[20] = 3
        kw[21] = 3
        kw[22] = 3
        kw[23] = 3
        kw[24] = 3
        kw[25] = 2
        
        p = 0
        s = [1,2,1,2,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
    
        # Input is a 300x400x3 size image
        x = tf.reshape(x, shape=[-1, self.imageHeight, self.imageWidth, 3])
    
        conv0 = tf.layers.conv2d(x, 32, (kh[0], kw[0]), s[0], activation=tf.keras.layers.LeakyReLU(0.01), kernel_initializer = initializer, 
                                         bias_initializer = initializer)
        
        mp1 = tf.layers.max_pooling2d(conv0, (kh[1], kw[1]), s[1])
                                    
        conv2 = tf.layers.conv2d(mp1, 64, (kh[2], kw[2]), s[2], activation=tf.keras.layers.LeakyReLU(0.01), kernel_initializer = initializer, bias_initializer = initializer)
        
        mp3 = tf.layers.max_pooling2d(conv2, (kh[3], kw[3]) ,s[3])
        
        conv4 = tf.layers.conv2d(mp3, 128,  (kh[4], kw[4]), s[4], activation=tf.keras.layers.LeakyReLU(0.01), kernel_initializer = initializer,
                                    bias_initializer = initializer)
        
        conv5 = tf.layers.conv2d(conv4, 64,  (kh[5], kw[5]), s[5], activation=tf.keras.layers.LeakyReLU(0.01), kernel_initializer = initializer,
                                    bias_initializer = initializer)
        
        conv6 = tf.layers.conv2d(conv5, 128,  (kh[6], kw[6]), s[6], activation=tf.keras.layers.LeakyReLU(0.01), kernel_initializer = initializer,
                                    bias_initializer = initializer)
        
        mp7 = tf.layers.max_pooling2d(conv6, (kh[7], kw[7]), s[7])
        
        
        conv8 = tf.layers.conv2d(mp7, 256, (kh[8], kw[8]), s[8],activation=tf.keras.layers.LeakyReLU(0.01), kernel_initializer = initializer,
                                    bias_initializer = initializer)
        
        conv9 = tf.layers.conv2d(conv8, 128, (kh[9], kw[9]), s[9],activation=tf.keras.layers.LeakyReLU(0.01), kernel_initializer = initializer,
                                    bias_initializer = initializer)
        
        conv10 = tf.layers.conv2d(conv9, 256, (kh[10], kw[10]), s[10], activation=tf.keras.layers.LeakyReLU(0.01), kernel_initializer = initializer,
                                    bias_initializer = initializer)
        
        mp11 = tf.layers.max_pooling2d(conv10, (kh[11], kw[11]), s[11])
        
        conv12 = tf.layers.conv2d(mp11, 512, (kh[12], kw[12]), s[12],activation=tf.keras.layers.LeakyReLU(0.01), kernel_initializer = initializer,
                                    bias_initializer = initializer)
        
        conv13 = tf.layers.conv2d(conv12, 256, (kh[13], kw[13]), s[13], activation=tf.keras.layers.LeakyReLU(0.01), kernel_initializer = initializer,
                                    bias_initializer = initializer)
        
        conv14 = tf.layers.conv2d(conv13, 512, (kh[14], kw[14]), s[14], activation=tf.keras.layers.LeakyReLU(0.01), kernel_initializer = initializer,
                                    bias_initializer = initializer)
        
        conv15 = tf.layers.conv2d(conv14, 256, (kh[15], kw[15]), s[15], activation=tf.keras.layers.LeakyReLU(0.01), kernel_initializer = initializer,
                                    bias_initializer = initializer)
        
        conv16 = tf.layers.conv2d(conv15, 512, (kh[16], kw[16]),  s[16],activation=tf.keras.layers.LeakyReLU(0.01), kernel_initializer = initializer,
                                    bias_initializer = initializer)
        
        mp17 = tf.layers.max_pooling2d(conv16, (kh[17], kw[17]), s[17])
        
        conv18 = tf.layers.conv2d(mp17, 1024, (kh[18], kw[18]), s[18], activation=tf.keras.layers.LeakyReLU(0.01), kernel_initializer = initializer,
                                    bias_initializer = initializer)
        
        conv19 = tf.layers.conv2d(conv18, 512, (kh[19], kw[19]), s[19],activation=tf.keras.layers.LeakyReLU(0.01), kernel_initializer = initializer,
                                    bias_initializer = initializer)
        
        conv20 = tf.layers.conv2d(conv19, 1024, (kh[20], kw[20]), s[20], activation=tf.keras.layers.LeakyReLU(0.01), kernel_initializer = initializer,
                                    bias_initializer = initializer)
        
        conv21 = tf.layers.conv2d(conv20, 512, (kh[21], kw[21]), s[21],activation=tf.keras.layers.LeakyReLU(0.01), kernel_initializer = initializer,
                                    bias_initializer = initializer)
        
        conv22 = tf.layers.conv2d(conv21, 1024, (kh[22], kw[22]), s[22],activation=tf.keras.layers.LeakyReLU(0.01), kernel_initializer = initializer,
                                    bias_initializer = initializer)
        
        conv23 = tf.layers.conv2d(conv22, 1024, (kh[23], kw[23]), s[23],activation=tf.keras.layers.LeakyReLU(0.01), kernel_initializer = initializer,
                                    bias_initializer = initializer)
        
        conv24 = tf.layers.conv2d(conv23, 1024, (kh[24], kw[24]), s[24],activation=tf.keras.layers.LeakyReLU(0.01), kernel_initializer = initializer,
                                    bias_initializer = initializer)
        
        conv25 = tf.layers.conv2d(conv24, 64, (kh[25], kw[25]), s[25],activation=tf.keras.layers.LeakyReLU(0.01), kernel_initializer = initializer,
                                    bias_initializer = initializer)
        
        conv26 = tf.layers.conv2d(conv25, 1024, (kh[26], kw[26]), s[26],activation=tf.keras.layers.LeakyReLU(0.01), kernel_initializer = initializer,
                                    bias_initializer = initializer)
        
        conv27 = tf.layers.conv2d(conv26, 1,  (kh[27], kw[27]), s[27],activation=tf.sigmoid, kernel_initializer = initializer,
                                    bias_initializer = initializer)
    
    
        # Convolution Layer with 15 filters and a kernel size of 5
        # Output: 296x396x15
        '''conv1 = tf.layers.conv2d(x, 15, 5, activation=tf.keras.layers.LeakyReLU(0.01), kernel_initializer = initializer,
                                    bias_initializer = initializer)
    
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        # Output: 148x198x15
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
    
        # Convolution Layer with 15 filters and a kernel size of 3
        # Output: 146x196x15
        conv2 = tf.layers.conv2d(conv1, 15, 3, activation=tf.keras.layers.LeakyReLU(0.01), kernel_initializer =  initializer,
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
        conv4 = tf.layers.conv2d(conv3, 2000, 1, activation=None, kernel_initializer = initializer,
                                    bias_initializer = initializer)
    
        # Output: 20x20x1000
        conv5 = tf.layers.conv2d(conv4, 2000, 1, activation=None, kernel_initializer = initializer,
                                    bias_initializer = initializer)
    
        # Output: 20x20x1
        conv6 = tf.layers.conv2d(conv5, 1, 1, activation = tf.sigmoid, kernel_initializer = initializer,
                                    bias_initializer = initializer)
    
        # Applies dropout to output
        out = tf.layers.dropout(conv6, rate=self.drop_prob, training=self.is_train)'''
    
        # out = tf.reshape(conv6, [-1,1])
    
        return conv27
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
                
                batch_idxs = self.numTrain//self.batch_size # config.batch_size
                
                print('Number mini-batch feeds required: ', batch_idxs)
                
                for idx in range(0, batch_idxs):
                    batch_images = self.shuffled_images[0][idx * self.batch_size : (idx + 1) * self.batch_size]
                    print(np.shape(batch_images))
                    batch_labels = self.shuffled_label_matrix[0][idx * self.batch_size : (idx + 1) * self.batch_size]
                    print(np.shape(batch_labels))
                    counter += 1
                    err, __, prd = self.sess.run([self.loss, self.train_op, self.pred], feed_dict={self.images: batch_images, self.labels: batch_labels})
                    
                
    
                    if counter % 10 == 0:
                        print("Epoch: ", (ep+1), " Step: ", counter, " Time: ", (time.time()-time_), " Loss: ", err)
                        #print("Prediction: ", prd)
                        #print(label_[1] - self.pred.eval({self.images: input_})[1],'loss:]',err)
                    if counter % 500 == 0:
                        self.save(config.checkpoint_dir, counter)
        # Test
        else:
            print("Now Start Testing...")
            
            for i in range(18):
                print('Test Image: ', i)
                print('Result:')
                result = self.pred.eval({self.images: np.expand_dims(self.shuffled_images[1][i], 0)})
                print(result)
                labelMat = (result > 0.5).astype(np.float32)
                print('Threshold Filtered Result')
                print(labelMat)
                print('Error: ' ,np.linalg.norm((labelMat-self.shuffled_label_matrix[1][i]), ord=None))
                
                
    
    
    
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

