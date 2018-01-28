'''

TODO


-> Add flags
    -> Is_Training
    -> Save_Model

'''

'''

Main File Pseudo Code (Logical Flow) 

-> Initialize are you training or only forward propagation
(Xiao Lei)
-> If training
    -> Initialize neural network accordingly
    -> Load training data (data set pixels and output matrix)
    -> Train the network
    -> Save model on every iteration
(Anish)
-> If not training
    -> Initialize neural network accordingly
    -> Load the model
    -> Load the perception screen
    -> While True: (eventually we can have key press to terminate loop)
        -> Get screenshot pixels
        -> Run neural network
        -> Feed predictions into perception screen
    
'''

import tensorflow as tf
import numpy as np
from CNN_Model import CNN_Model

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean("is_train", True, "if training")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory")
flags.DEFINE_string("test_dir", "", "test images directory")

def main(_): #?
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = CNN_Model(sess,
                      is_train = FLAGS.is_train,
                      test_dir = FLAGS.test_dir
                      )
        model.train(FLAGS)
        
if __name__=='__main__':
    tf.app.run() # parse the command argument , the call the main function