# ===================================== IMPORTS ===================================== #

from PIL import Image
import os
import tensorflow as tf

# ================================== LOCAL IMPORTS ================================== #

import py_pixel as p
import neural_network as nn
import data_set as ds

def valid_type(var):

    if (str(type(var)) == r"<class 'list'>") or \
    (str(type(var)) == r"<class 'tuple'>") or \
    (str(type(var)) == r"<class 'numpy.ndarray'>"):
        return True
    else:
        return False

if __name__ == '__main__':

    # mode = tf.placeholder(tf.string,shape=[3,3,3], name='mode')
    # bias_var = tf.constant(0.1, shape=[1,1])
    #
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())

    frame_size = [128,128]
    frame_stride_size = [128,128]
    num_layers = 7
    num_conv_layers = 7
    num_fc_layers = 0
    f_matrix = [5,5,5,5,5,5,104]#[5, 5, 120]  # TO DO: SEPARATE FILTER SIZES FOR X AND Y
    z_matrix = [10,10,10,10,10,10,1]#[32, 64, 6]
    p_matrix = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
    s_matrix = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]  # strides for entire image
    ps_matrix = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
    p_type = 'MAX'
    a_type = 'SIG'

    image_num = 1
    image = Image.open(ds.get_image(image_num))
    full_image_w, full_image_l, full_image_h = p.get_image_size(image)
    img_data = ds.get_image_data(image_num, [full_image_w, full_image_l, full_image_h], frame_size)
    frames, num_frames_x, num_frames_y = p.get_frames(image, frame_size, frame_stride_size)
    image_w, image_l, image_h = p.get_image_size(frames[0][0])

    valid_network = nn.check_nn_config(layers=num_layers,
                                       conv_layers=num_conv_layers,
                                       fc_layers = 0,
                                       conv_z_matrix=z_matrix,
                                       fil_size_matrix=f_matrix,
                                       pooling_matrix=p_matrix,
                                       stride_matrix=s_matrix,
                                       pool_stride_matrix=ps_matrix,
                                       pool_type=p_type,
                                       activation_type=a_type,
                                       input_x=image_w,
                                       input_y=image_l,
                                       input_z=image_h)


    print("\nNetwork Valid? == {}".format(valid_network))

    if valid_network == 1:

        output_shape = nn.get_output_shape(layers=num_layers,
                                        conv_layers=num_conv_layers,
                                        fc_layers = num_fc_layers,
                                        conv_z_matrix=z_matrix,
                                        fil_size_matrix=f_matrix,
                                        pooling_matrix=p_matrix,
                                        stride_matrix=s_matrix,
                                        pool_stride_matrix=ps_matrix,
                                        pool_type=p_type,
                                        activation_type=a_type,
                                        input_x=image_w,
                                        input_y=image_l,
                                        input_z=image_h)

        minion_neural_network = nn.conv_net(layers=num_layers,
                                            conv_layers=num_conv_layers,
                                            fc_layers = num_fc_layers,
                                            conv_z_matrix=z_matrix,
                                            fil_size_matrix=f_matrix,
                                            pooling_matrix=p_matrix,
                                            stride_matrix=s_matrix,
                                            pool_stride_matrix=ps_matrix,
                                            pool_type=p_type,
                                            activation_type=a_type,
                                            input_x=image_w,
                                            input_y=image_l,
                                            input_z=image_h,
                                            output_shape=output_shape)


        print("\n--------------------- Execution Start ---------------------\n")

        accuracy = 0
        accuracy_tot = 0
        accuracy_cnt = 0
        accuracy_same_cnt = 0
        prev_accuracy = 0

        num_images = ds.get_num_processed_images()
        images_processed = 0
        gradient_val = 1e-12
        CF = 0

        while (images_processed < num_images) or (accuracy < 0.95):

            print("Getting new image.")
            if images_processed == num_images:
                images_processed = 0
            else:
                image_num = images_processed + 1
            image = Image.open(ds.get_image(image_num))

            print("Getting image data.")
            full_image_w, full_image_l, full_image_h = p.get_image_size(image)
            img_data = ds.get_image_data(image_num, [full_image_w, full_image_l, full_image_h], frame_size)
            frames, num_frames_x, num_frames_y = p.get_frames(image, frame_size, frame_stride_size)

            good_data_flag = False
            data_found_flag = 1
            used_frame = []

            while (accuracy == 1.0) or ((accuracy < 0.95) and (data_found_flag == 1)):

                good_data_flag = not good_data_flag
                data_found_flag = 0

                if good_data_flag == True:
                    for x in range(num_frames_x):
                        for y in range(num_frames_y):
                            used_flag = 0
                            for uxy in used_frame:
                                if (uxy == (str(x)+str(y))):
                                    used_flag = 1
                            if (used_flag == 0) and (img_data[x][y][4] == 1):
                                    current_frame = frames[x][y]
                                    current_frame_x = x
                                    current_frame_y = y
                                    data_found_flag = 1
                elif good_data_flag == False:
                    for x in range(num_frames_x):
                        for y in range(num_frames_y):
                            used_flag = 0
                            for uxy in used_frame:
                                if (uxy == (str(x)+str(y))):
                                    used_flag = 1
                            if (used_flag == 0) and (img_data[x][y][4] == 0):
                                    current_frame = frames[x][y]
                                    current_frame_x = x
                                    current_frame_y = y
                                    data_found_flag = 1

                if data_found_flag == 1:

                    used_frame.append(str(current_frame_x) + str(current_frame_y))
                    #print("{} {}".format(used_frame,(str(current_frame_x) + str(current_frame_y))))

                    pixels = p.get_pixels(current_frame)

                    nn_input = img_data[current_frame_x][current_frame_y][5]
                    #print(nn_input)

                    #A_c, NN_c, PE, SE, CE, PRE, CF, OC, A_pos_x, A_pos_y, NN_pos_x, NN_pox_y, G, NN_c_unformatted, var_grad  = minion_neural_network.execute(pixels,nn_input,gradient_val)
                    A_c, NN_c, CF, BCE, OC, var_grad, CW = minion_neural_network.execute(pixels,[nn_input],gradient_val, CF)

                    if OC == True:#OC[0] == True:
                        accuracy_tot = accuracy_tot + (1 == OC)

                    accuracy_cnt+=1
                    accuracy = float(accuracy_tot)/accuracy_cnt
                    if ('%.2f' % prev_accuracy)  == ('%.2f' % accuracy):
                        accuracy_same_cnt+=1
                    else:
                        accuracy_same_cnt = 0

                    if (accuracy_same_cnt >= 10) and (gradient_val < 1e-3):
                        gradient_val = gradient_val#float(gradient_val*2)
                        accuracy_same_cnt = 0
                    prev_accuracy = accuracy

                    # Printing Specfic Gradients

                    # Printing Gradient Matrix Data
                    # for grad_list in var_grad:
                    #     grad_print = grad_list
                    #     string = ''
                    #     while (str(type(grad_print)) == r"<class 'list'>") or \
                    #           (str(type(grad_print)) == r"<class 'tuple'>") or \
                    #           (str(type(grad_print)) == r"<class 'numpy.ndarray'>"):
                    #         string = string + '[' + str(len(grad_print)) + ']'
                    #         grad_print = grad_print[0]
                    #     print(string)
                    # input()

                    # Printing Network Data
                    # print("A_c: {} , NN_c: {}, PE: {}, SE: {}, CE: {}, PRE: {}, CF: {}, OC: {}, Acc: {}, A_pos: ({},{}), NN_pos: ({},{}), Image_Flag: {}, X: {}. Y: {}, ASC: {}, GV: {}, NN_c_un: {}, Var Grad: {}"
                    #       .format('%.0f' % A_c, '%.2f' % NN_c, '%.2E' % PE, '%.2E' % SE, '%.2E' % CE, '%.2E' % PRE, '%.2E' % CF, OC,'%.7f' % accuracy,
                    #               '%.2f' % A_pos_x, '%.2f' % A_pos_y, '%.2f' % NN_pos_x, '%.2f' % NN_pox_y, good_data_flag, current_frame_x, current_frame_y,accuracy_same_cnt, gradient_val, NN_c_unformatted, var_grad[4][0][0][0][0] ))

                    # Printing Network Data
                    print("A_c: {} , NN_c: {}, CE: {}, CF: {}, OC: {}, Acc: {}, ASC: {}, GV: {}, VG: {}"
                          .format('%.5f' % A_c, '%.5f' % NN_c,'%.2E' % BCE, '%.2E' % CF, '%.2E' % OC,
                                  '%.5F' % accuracy, '%.2F' % accuracy_same_cnt, '%.2E' % gradient_val, CF ))
                    #print("CW: {}".format(CW))

                    # curr_grad = var_grad
                    # for index in len(curr_grad):
                    #     while valid_type(curr_grad) == True:
                    #

                    #print(var_grad)
                    #input('Waiting for input...')
                    #print("-----------------------------------------------------------------------------------------------------------------------------------------------")
                    # print(var_grad[4][1][0][0][0][0])
                    # print(var_grad[4][0][0][0][0][1])
                    # print(var_grad[4][1][0][0][0][1])
                    # print(var_grad[4][0][0][0][0][2])
                    # print(var_grad[4][1][0][0][0][2])
                    # print(var_grad[4][0][0][0][0][3])
                    # print(var_grad[4][1][0][0][0][3])
                    # print(var_grad[4][0][0][0][0][4])
                    # print(var_grad[4][1][0][0][0][4])
                    # print(var_grad[4][0][0][0][0][5])
                    # print(var_grad[4][1][0][0][0][5])

            print("Done processing frames...")

            print('Num Images: {} , Images Processed: {}'.format(num_images, images_processed))
            images_processed+=1

    else:
        print("\nInvalid network")
