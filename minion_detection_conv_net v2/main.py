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

'''
Shuffles two lists together matching the element's
respective position in the 2 lists.

l1, l2:  the lists to be shuffled together
a, b: the shuffled lists l1 and l2
'''
def shuffleTwoLists(l1, l2):
    
    lA = cp.deepcopy(l1)
    lB = cp.deepcopy(l2)
    c = list(zip(lA,lB))
    rand.shuffle(c)
    
    # Unzips the list of tuples
    a, b = zip(*c)
    return a,b

'''
Given x, returns a matrix of shape x.
This matrix's particular element is 1
iff x's corresponding element >= 0.5

x: the input matrix
out: matrix to be returned as described above
'''
def binary_activation(x):
    
    num = tf.zeros(shape=tf.shape(x), dtype=tf.float32)+0.5
    cond = tf.less(x, num)
    out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))

    return out

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
    pred_classes = binary_activation(logits_train)
    
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
    
    num_frames = 60
    
    # Training Parameters
    learning_rate = 1e-5
    num_steps = 2000
    batch_size = 5
    
    # Network Parameters
    num_classes = 1 # MNIST total classes (0-9 digits)
    dropout = 0.25 # Dropout, probability to drop a unit
    
    # Specifies number of frames for training, cross_validation, and evalutation
    training_frames = 25
    testing_frames = 25
    evaluation_frames = 10
    
    # Calculates split positions for each type of data
    splitInfo = [ training_frames, testing_frames, evaluation_frames]
    splitInfoNP = np.array(splitInfo).astype(np.int32)
    cumSplitInfo = np.cumsum(splitInfoNP, axis=0, dtype=np.int32)
    
    # Gets number of images in the data set
    num_images = ds.get_num_processed_images()
        
    print('Total number of images: ', num_images)
    print('Total number of frames: ', num_frames)
    print('Batch size: ', batch_size)
    print('Learning Rate: ', learning_rate)
    
    
    # Extracts num_frames frames from image data set
    framesInput = ds.get_frames(num_frames)
    
    # Shuffles our frames
    frames, lbls = shuffleTwoLists(framesInput[0], framesInput[1])
    
    # Converts frames and labels to np array
    framesNP = np.array(frames).astype(np.float32)
    lblsNP = np.array(lbls).reshape([-1,1]).astype(np.float32)
    
    # Splits the frames and labels based on cumSplitInfo
    splitFrames = np.split(framesNP, cumSplitInfo, 0)
    splitLbls = np.split(lblsNP, cumSplitInfo, 0)
    
    # Assigns frames and labels arrays
    trainFramesNP = splitFrames[0]
    testFramesNP = splitFrames[1]
    evalFramesNP = splitFrames[2]
    
    trainLblsNP = splitLbls[0]
    testLblsNP = splitLbls[1]
    evalLblsNP = splitLbls[2]   
    
    print('Train Data shape: ', np.shape(trainFramesNP))
    print('Test Data shape: ', np.shape(testFramesNP))
    print('Eval Data shape: ', np.shape(evalFramesNP))
    print('Train Label shape: ', np.shape(trainLblsNP))
    print('Test Label shape: ', np.shape(testLblsNP))
    print('Eval Label shape: ', np.shape(evalLblsNP))
    
    print("\n--------------------- Execution Start ---------------------\n")
        
    # Build the Estimator
    model = tf.estimator.Estimator(model_fn)
    
    #framesInputNP = np.array(framesInput[0]).astype(np.float32)
    #labels = np.array(framesInput[1]).reshape([-1,1]).astype(np.float32)
    
    # Define the input function for training
    input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': trainFramesNP}, y=trainLblsNP,
    batch_size=batch_size, num_epochs=None, shuffle=True)
    
    # Train the Model
    model.train(input_fn, steps=num_steps)
    

    # Evaluate the Model
    # Define the input function for evaluating
    input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': testFramesNP}, y=testLblsNP,
    batch_size=batch_size, shuffle=False)
    
    # Use the Estimator 'evaluate' method
    e = model.evaluate(input_fn)

    print("Testing Accuracy:", e['accuracy'])





