# ===================================== IMPORTS ===================================== #

from __future__ import division, print_function, absolute_import
from PIL import Image
import os
import tensorflow as tf
import numpy as np
import math
import copy as cp
import random as rand
import CNNModel as cnn

# ================================== LOCAL IMPORTS ================================== #

import py_pixel as p
import data_set as ds

def valid_type(var):

    if (str(type(var)) == r"<class 'list'>") or \
    (str(type(var)) == r"<class 'tuple'>") or \
    (str(type(var)) == r"<class 'numpy.ndarray'>"):
        return True
    else:
        return False

def shuffleTwoLists(l1, l2):
    
    lA = cp.deepcopy(l1)
    lB = cp.deepcopy(l2)
    c = list(zip(lA,lB))
    rand.shuffle(c)
    a, b = zip(*c)
    return a,b
    
# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = cnn.conv_net(features, num_classes, dropout, reuse=False,
                            is_training=True)
    logits_test = cnn.conv_net(features, num_classes, dropout, reuse=True,
                           is_training=False)
    
    print('Predictions:')
    print(logits_train)
    print('Labels:')
    print(labels)
    # Predictions
    #pred_classes = tf.argmax(logits_test, axis=1)
    pred_classes = logits_train
    pred_probas = tf.nn.softmax(logits_test)
    
    print('Prediction Shape:', np.shape(pred_classes))
    print('Labels Shape:', np.shape(labels))

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)
    
    
    # Define loss and optimizer
    loss_op = -tf.reduce_mean(labels*tf.log(logits_train+1e-10) + (1-labels)*tf.log(1-(logits_train+1e-10)))
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs
if __name__ == '__main__':

    # mode = tf.placeholder(tf.string,shape=[3,3,3], name='mode')
    # bias_var = tf.constant(0.1, shape=[1,1])
    #
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    
    num_frames = 30
    
    # Training Parameters
    learning_rate = 1e-9
    num_steps = 10000
    batch_size = 10

    # Network Parameters
    num_input = 30 # MNIST data input (img shape: 28*28)
    num_classes = 1 # MNIST total classes (0-9 digits)
    dropout = 0.25 # Dropout, probability to drop a unit
    
    # Gets number of images in the data set
    num_images = ds.get_num_processed_images()
    images_processed = 0
        
    print('Total number of images: ', num_images)
        
        
    # cost function
    # CF = 0
        
    # Extracts num_frames frames from image data set
    framesInput = ds.get_frames(num_frames)
    
    print("\n--------------------- Execution Start ---------------------\n")
        
    # Build the Estimator
    model = tf.estimator.Estimator(model_fn)
    
    framesInputNP = np.array(framesInput[0]).astype(np.float32)
    labels = np.array(framesInput[1]).reshape([-1,1]).astype(np.float32)
    
    print('Data shape: ', np.shape(framesInputNP))
    print('Label shape: ', np.shape(labels))
    
    # Define the input function for training
    input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': framesInputNP}, y=labels,
    batch_size=batch_size, num_epochs=None, shuffle=True)
    
    # Train the Model
    model.train(input_fn, steps=num_steps)
    

    # Evaluate the Model
    # Define the input function for evaluating
    '''input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images}, y=mnist.test.labels,
    batch_size=batch_size, shuffle=False)
    # Use the Estimator 'evaluate' method
    e = model.evaluate(input_fn)'''

    #print("Testing Accuracy:", e['accuracy'])





