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

# ================================ LOCAL IMPORTS ================================ #

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
                ):
        self.sess = sess
        self.is_train = is_train
        self.test_dir = test_dir

        # Specifies image parameters
        self.imageHeight = 344
        self.imageWidth = 258
        self.numPartsX = 20 #TODO check how big bounding boxes are in general
        self.numPartsY = 15

        self.images = tf.placeholder(tf.float32,[None,self.imageHeight,self.imageWidth,3],name='images')
        self.labels = tf.placeholder(tf.float32,[None,self.numPartsY,self.numPartsX,1],name='labels')

        # Specifies training parameters
        self.epoch = 400
        self.batch_size = 5
        self.learning_rate = 1e-5
        self.drop_prob = 0.25

        # Creates the model
        self.pred = self.model
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
        self.saver = tf.train.Saver()

# Create the network architecture (we feed this into the train function)
def conv_net(self):

    # TF Estimator input is a dict, in case of multiple inputs
    x = x_dict['images']

    initializer = tf.contrib.layers.xavier_initializer_conv2d()

    # Input is a 344 258 x3 size image
    x = tf.reshape(x, shape=[-1, self.imageHeight, self.imageWidth, 3])

    # Convolution Layer with 15 filters and a kernel size of 5
    # Output: 340x254x15
    conv1 = tf.layers.conv2d(x, 15, 5, activation=tf.keras.layers.LeakyReLU(0.01), kernel_initializer = initializer,
                                bias_initializer = initializer)

    # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
    # Output: 170 x 127 x 15
    conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

    # Convolution Layer with 15 filters and a kernel size of 3
    # Output: 168x125x15
    conv2 = tf.layers.conv2d(conv1, 15, 3, activation=tf.keras.layers.LeakyReLU(0.01), kernel_initializer = initializer,
                                bias_initializer = initializer)

    # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
    # Output: 84x62x15
    conv2 = tf.layers.max_pooling2d(conv2, 2, 2)


    # Output: 82x60x15
    conv3 = tf.layers.conv2d(conv2, 15, (42, 30), activation=tf.keras.layers.LeakyReLU(0.01), kernel_inializer = initializer,
                                bias_initializer = initializer)

    # Output: 41x31x15
    conv3 = tf.layers.max_pooling2d(conv3, 2, 2)

    # Fully connected layers using conv layers
    # Output: 20x15x1000
    # Filter size: 116x90
    conv4 = tf.layers.conv2d(conv3, 1000, 1, activation=None, kernel_initializer = initializer,
                                bias_initializer = initializer)

    # Output: 20x15x1000
    conv5 = tf.layers.conv2d(conv4, 1000, 1, activation=None, kernel_initializer = initializer,
                                bias_initializer = initializer)

    # Output: 20x15x1
    conv6 = tf.layers.conv2d(conv5, 1, 1, activation = tf.sigmoid, kernel_initializer = initializer,
                                bias_initializer = initializer)

    # Applies dropout to output
    out = tf.layers.dropout(conv6, rate=self.drop_prob, training=self.is_train)

    # out = tf.reshape(conv6, [-1,1])

    return out

def train(self, config):

    # NOTE : if train, the nx, ny are ingnored
    input_setup(config)

    data_dir = checkpoint_dir(config)

    images =  ds.load_images()
    label_matrix = dlb.load_matrix()
    shuffled_images, shuffled_label_matrix = prepare_data_set(images,label_matrix)

    print(shuffled_images.shape, shuffled_label_matrix.shape)

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
            batch_idxs = len(shuffles_images) // config.batch_size
            for idx in range(0, batch_idxs):
                batch_images = shuffled_images[idx * self.batch_size : (idx + 1) * config.batch_size]
                batch_labels = shuffled_label_matrix[idx * self.batch_size : (idx + 1) * config.batch_size]
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
        result = self.pred.eval({self.images: shuffled_images[1].reshape(1, self.imageHeight, self.imageWidth, 3)})
        x = np.squeeze(result)

        print(x.shape)



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

