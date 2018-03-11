
import tensorflow as tf
import numpy as np
import sys
import time
import win_unicode_console

win_unicode_console.enable()

sys.path.insert(0, r'C:\Users\OM\Desktop\LOL-Autolane\LOL-Autolane\v4\Neural_Network')
from CNN_Model import CNN_Model

sys.path.insert(0, r'C:\Users\OM\Desktop\LOL-Autolane\LOL-Autolane\v4\Perception')
import perception as p
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

sys.path.insert(0, r'C:\Users\OM\Desktop\LOL-Autolane\LOL-Autolane\v4\Screen_Capture')
import screen_capture as sc

sys.path.insert(0, r'C:\Users\OM\Desktop\LOL-Autolane\LOL-Autolane\v4\Data_Set')
import data_set as ds

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean("is_train", False, "if training")
flags.DEFINE_string("checkpoint_dir", "checkpoint", r"C:\Users\OM\Desktop\LOL-Autolane\LOL-Autolane\v4\Checkpoint")
flags.DEFINE_string("test_dir", "", "test images directory")
flags.DEFINE_boolean("is_realTime", True, "real time running")

global processed_data_path
processed_data_path = r"C:\Users\OM\Desktop\processed_dataset"

def get_image_path(image_num):
    return(processed_data_path + r'\processed_{}'.format(image_num) + '.png')

if __name__=='__main__':
    #tf.app.run() # parse the command argument , the call the main function
    
    screen_size_x = 400
    screen_size_y = 300

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        ps = p.perception_screen(screen_size_x, screen_size_y)
        model = CNN_Model(sess,
                      is_train = FLAGS.is_train,
                      test_dir = FLAGS.test_dir,
                      is_realTime = FLAGS.is_realTime
                      )
        if FLAGS.is_realTime:
            # model.runRealTime(FLAGS)
            model.runRealTime(FLAGS)

            rgb = sc.initialize_rgb_array()

            while True:
            #     print('Check3')
            #     curr_img = Image.open(get_image_path(47))
            #     second_img = sc.get_screen_capture(rgb, return_image=True)[1]
            #     print('Check4')
            #     print('Conv Net Input:')
            #     print(len(ds.load_images(real_time_flag=True, image=curr_img)))
            #     print(len(ds.load_images(real_time_flag=True, image=curr_img)[0]))
            #     print(len(ds.load_images(real_time_flag=True, image=curr_img)[0][0]))
            #     print(ds.load_images(real_time_flag = True,image=curr_img)[0][0][0])
            #     print(ds.load_images(real_time_flag=True, image=curr_img)[0][50][50])
            #
            #     print("Image Diff:")
            #     print(ds.load_images(real_time_flag=True, image=curr_img)[0][0][0] -
            #           ds.load_images(real_time_flag=True, image=second_img)[0][0][0])
            #     print(ds.load_images(real_time_flag = True,image=curr_img)-
            #           ds.load_images(real_time_flag=True, image=second_img))
            #
            #     result = model.pred.eval({model.images: ds.load_images(real_time_flag = True,image=curr_img)})
            #     print('Check5')
            #
            # # for i in range(0,10):
            # #
            # #     print('Conv Net Input:')
            # #     print(len(model.shuffled_images[1][i]))
            # #     print(len(model.shuffled_images[1][i][0]))
            # #     print(model.shuffled_images[1][i][0][0])
            # #     print(model.shuffled_images[1][i][50][50])
            # #
            # #
            # #     result = model.pred.eval({model.images: np.expand_dims(model.shuffled_images[1][i], 0)})
            #
            #     labelMat = (result > 0.5).astype(np.integer)
            #     labelMat = np.squeeze(labelMat,axis=0)
            #     labelMat = np.squeeze(labelMat, axis=2)
            #
            #     print('Threshold Filtered Result')
            #     print(labelMat)
            #
            #     # correctAnswer = model.shuffled_label_matrix[1][i]
            #     # correctAnswer = np.squeeze(correctAnswer,axis=2)
            #     #
            #     # print('Correct Answer:')
            #     # print(correctAnswer)
            #     #
            #     # print('Difference:')
            #     # print(labelMat - correctAnswer.astype(np.integer))
            #
            #     curr_img.show()
            #     #plt.imshow(curr_img)
            #     #plt.imshow(model.shuffled_images[1][i])
            #     #plt.show()

                #print('Check3')
                curr_img = Image.open(get_image_path(47))
                second_img = sc.get_screen_capture(rgb, return_image=True)[1]
                #print('Check4')
                #print('Conv Net Input:')
                # print(len(ds.load_images(real_time_flag=True, image=curr_img)))
                # print(len(ds.load_images(real_time_flag=True, image=curr_img)[0]))
                # print(len(ds.load_images(real_time_flag=True, image=curr_img)[0][0]))
                # print(ds.load_images(real_time_flag=True, image=curr_img)[0][0][0])
                # print(ds.load_images(real_time_flag=True, image=curr_img)[0][50][50])

                result = model.pred.eval({model.images: ds.load_images(real_time_flag=True, image=second_img)})
                #print('Check5')

                # for i in range(0,10):
                #
                #     print('Conv Net Input:')
                #     print(len(model.shuffled_images[1][i]))
                #     print(len(model.shuffled_images[1][i][0]))
                #     print(model.shuffled_images[1][i][0][0])
                #     print(model.shuffled_images[1][i][50][50])
                #
                #
                #     result = model.pred.eval({model.images: np.expand_dims(model.shuffled_images[1][i], 0)})

                labelMat = (result > 0.5).astype(np.integer)
                labelMat = np.squeeze(labelMat, axis=0)
                labelMat = np.squeeze(labelMat, axis=2)

                # print('Threshold Filtered Result')
                print(labelMat)
                ps.draw_matrix(labelMat.transpose())

                # correctAnswer = model.shuffled_label_matrix[1][i]
                # correctAnswer = np.squeeze(correctAnswer,axis=2)
                #
                # print('Correct Answer:')
                # print(correctAnswer)
                #
                # print('Difference:')
                # print(labelMat - correctAnswer.astype(np.integer))

                #second_img.show()
                #input()
                #plt.imshow(second_img)
                #plt.imshow(model.shuffled_images[1][i])
                #plt.show()
                #time.sleep(1)
        else:
            model.train(FLAGS)

