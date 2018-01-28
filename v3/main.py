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
import sys
import time

sys.path.insert(0, 'C:\\Users\\HP_OWNER\\Desktop\\LOL-Autolane\\v3\\Neural_Network')
from CNN_Model import CNN_Model

sys.path.insert(0, r'C:\\Users\\HP_OWNER\\Desktop\\LOL-Autolane\\v3\\Perception')
import perception as p
from matplotlib import pyplot
import matplotlib.image as mpimg

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean("is_train", True, "if training")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory")
flags.DEFINE_string("test_dir", "", "test images directory")
flags.DEFINE_boolean("is_realTime", False, "real time running")

'''def main(_): #?
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = CNN_Model(sess,
                      is_train = FLAGS.is_train,
                      test_dir = FLAGS.test_dir
                      )
        model.train(FLAGS)'''
        
if __name__=='__main__':
    #tf.app.run() # parse the command argument , the call the main function
    
    screen_size_x = 344
    screen_size_y = 258
    perception_screen = p.perception_screen(screen_size_x, screen_size_y)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = CNN_Model(sess,
                      is_train = FLAGS.is_train,
                      test_dir = FLAGS.test_dir,
                      is_realTime = FLAGS.is_realTime
                      )
        if FLAGS.is_realTime:
            # model.runRealTime(FLAGS)
            model.runRealTime(FLAGS)
            
            for i in range(16):    
                result = model.pred.eval({model.images: np.expand_dims(model.shuffled_images[1][i], 0)})
                #print(result)
                labelMat = (result > 0.5).astype(np.integer)
                #print(np.shape(labelMat))
                labelMat = np.squeeze(labelMat,axis=0)
                labelMat = np.squeeze(labelMat, axis=2)
                #perception_screen.draw_matrix(labelMat)
                pyplot.imshow(model.shuffled_images[1][i])
                time.sleep(5)
            
        else:
            model.train(FLAGS)