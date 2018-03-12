# IMPORT
# =========================================
from Settings import *

win_unicode_console.enable()

# MAIN
# =========================================

if __name__=='__main__':

    # Setting up the GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        model = CNN.Model(sess)

        if (real_time_flag == True) and (train_flag == False) and (test_flag == False):

            rgb = sc.initialize_rgb_array()

            ps = perception.perception_screen(game_screen_x, game_screen_y)

            model.runRealTime()

            while True:

                pixels,img = sc.get_screen_capture(rgb)

                result = model.pred.eval({model.images: [pixels]})

                labelMat = (result > 0.5).astype(np.integer)
                labelMat = np.squeeze(labelMat, axis=0)
                labelMat = np.squeeze(labelMat, axis=2)

                print(labelMat)
                ps.draw_matrix(labelMat.transpose())

        elif (real_time_flag == False) and (train_flag == True) and (test_flag == False):

            model.train()

        elif (real_time_flag == False) and (train_flag == False) and (test_flag == True):

            model.runRealTime()

            for i in range(20, ds.get_num_processed_images()):

                img = Image.open(ds.get_image_path(i))
                img_pixels = sc.get_pixels(img)

                result = model.pred.eval({model.images: [img_pixels]})

                labelMat = (result > 0.5).astype(np.integer)
                labelMat = np.squeeze(labelMat, axis=0)
                labelMat = np.squeeze(labelMat, axis=2)

                print(labelMat)
                img.show()
                input()