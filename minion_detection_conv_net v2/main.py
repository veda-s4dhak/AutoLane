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
    
    # Specifies number of optimization steps
    epochs = 100
    
    # Specifies batch size (number of images per batch)
    bSize = 2
    
    # Specifies number of frames per image
    frameBatchSize = 6
    
    # Specifies the neural network architecture: from line 31 to 60
    frame_size = [128,128]
    
    # Stride size (not overlapping) shift by 128
    frame_stride_size = [128,128]
    num_layers = 7
    
    # no fully connected layer; total 7 conv layers
    num_conv_layers = 7
    num_fc_layers = 0
    
    # 1st filter 5x5 2nd filter 5x5 last filter: 104x104 
    f_matrix = [5,5,5,5,5,5,104]#[5, 5, 120]  # TO DO: SEPARATE FILTER SIZES FOR X AND Y
    
    # f_matrix = [[5,7],5,5,5,5,5,104] feed non square filter
    
    # first 5x5 filter 10 channels; last one 1 channel
    z_matrix = [10,10,10,10,10,10,1]#[32, 64, 6]
    
    # Pool matrix: size of pools for each layer
    p_matrix = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
    
    # stride matrix for each filter
    s_matrix = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]  # strides for entire image
    
    # pool stride matrix; standard pool size
    ps_matrix = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
    p_type = 'MAX'
    
    # Can try relu
    a_type = 'RELU'
    ##############################################
    
    # Line 64 to 69: Gets data on images
    image_num = 1
    image = Image.open(ds.get_image(image_num))
    full_image_w, full_image_l, full_image_h = p.get_image_size(image)
    img_data = ds.get_image_data(image_num, [full_image_w, full_image_l, full_image_h], frame_size)
    frames, num_frames_x, num_frames_y = p.get_frames(image, frame_size, frame_stride_size)
    image_w, image_l, image_h = p.get_image_size(frames[0][0])
    
    # Line 72 to 84: checks if neural network is valid
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

    # Prints if network is valid
    print("\nNetwork Valid? == {}".format(valid_network))

    if valid_network == 1:
        
        # Line 92 to 104: determines output shape of neural network
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
        
        # Line 107-120: initializes the neural network 
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
        
        # Line 126 - 130: initializes parameters to calculate accuracy
        accuracy = 0
        accuracy_tot = 0
        accuracy_cnt = 0
        accuracy_same_cnt = 0
        prev_accuracy = 0
        
        # Line 133- 136: initializes the while loop
        
        # Gets number of images in the data set
        num_images = ds.get_num_processed_images()
        images_processed = 0
        
        print('Total number of images: ', num_images)
        
        # learning rate
        gradient_val = 1e-12
        
        # cost function
        # CF = 0
        
        
        # quits when all images processed and accuracy reached 95%
        # while (images_processed < num_images) or (accuracy < 0.95):
        #while (images_processed < num_images):
        for i in range(epochs):
            
            # Line 148 - 153: getting the next image in the data set
            '''print("Getting new image.")
            if images_processed == num_images:
                images_processed = 0
            else:
                image_num = images_processed + 1
            image = Image.open(ds.get_image(image_num))

            # Line 156 - 159: getting data on the image; bounding boxes
            print("Getting image data.")
            
            # Gets dimension of image
            full_image_w, full_image_l, full_image_h = p.get_image_size(image)
            
            # Gets bounding box data
            img_data = ds.get_image_data(image_num, [full_image_w, full_image_l, full_image_h], frame_size)
            
            # Separates image into frames; separates image into grid
            frames, num_frames_x, num_frames_y = p.get_frames(image, frame_size, frame_stride_size)'''
            
            # Gets current batch
            # Structure of each element in batch list: [Images[i], fullImageDim[i], imageBoundData[i], frameData[i]]
            # Structure of each element in fullImageDim: [full_image_w, full_image_l, full_image_h]
            # Structure of each element in imageBoundData: img_data
            # Structure of each element in frameData: [frames, num_frames_x, num_frames_y]
            batch = ds.get_image_batch(batch_size=bSize)
            
            # quits when accruacy is not 1 and greater than 95% or no data processed
            # while (accuracy == 1.0) or ((accuracy < 0.95) and (data_found_flag == 1)):
            # Gets frameBatchSize frames
            
            # Initializes input batch to convnet
            # First list contains the frames
            # Second list contains the labels for the frames
            InputBatch = [[],[]]
            
            # Generates bSizexframeBatchSize frames to be feeded into convnet
            # Appends frame and its label to InputBatch
            for j in range(bSize):
                
                # accumulates all frames/grids processed in the image
                used_frame = []
                
                # true iff there we are feeding a frame that has a minion for that iteration
                good_data_flag = False
                
                # initializes image data for current image
                data_found_flag = 1
                frames = batch[j][3][0]
                num_frames_x = batch[j][3][1]
                num_frames_y = batch[j][3][2]
                img_data = batch[j][2]
                imgDimension = batch[j][1]
                full_image_w, full_image_l, full_image_h = imgDimension[0], imgDimension[1], imgDimension[2]
                
                for k in range(frameBatchSize):
                    good_data_flag = not good_data_flag
                
                    # initializes data found flag to 0
                    data_found_flag = 0
                
                    # finding a frame which has not been processed that has a minion
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
                                    
                    # finding a frame which has not been processed that does not have a minion
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
                                    
                    # processes data if frame we are looking for (has minion/not) is found; done otherwise
                    if data_found_flag == 1:
                    
                        # adding frame to used frame list
                        used_frame.append(str(current_frame_x) + str(current_frame_y))
                        #print("{} {}".format(used_frame,(str(current_frame_x) + str(current_frame_y))))
                    
                        # gets pixel data of current frame
                        pixels = p.get_pixels(current_frame)
                    
                        # image data contains all bounding boxes of the image
                        # curr_frame_x,y specifies a certain frame; 5th value is class
                        # assigns label to be feeded into nn
                        nn_input = img_data[current_frame_x][current_frame_y][5]
                        
                        # Appends new data to input batch
                        InputBatch[0].append(pixels)
                        InputBatch[1].append(nn_input)
                        
                        
                        
                        
                        #print(nn_input)
                        

            #A_c, NN_c, PE, SE, CE, PRE, CF, OC, A_pos_x, A_pos_y, NN_pos_x, NN_pox_y, G, NN_c_unformatted, var_grad  = minion_neural_network.execute(pixels,nn_input,gradient_val)
            # executes neural network; feeds in pixels of that frame(has minion/not); nn_input/assigned label; learning rate; cost function from previous iteration
            # A_c, NN_c, CF, E, OC, var_grad, CW = minion_neural_network.execute(pixels,[nn_input],gradient_val, CF)
            
            # gradient_val is the learning rate
            #A_c, NN_c, CF, E, OC, var_grad, CW = minion_neural_network.execute(InputBatch, gradient_val)
           
            A_c, NN_c, OC, CF, CW = minion_neural_network.execute(InputBatch, gradient_val)
            
            # calculates accuracy line 233-241
            '''if OC == True:#OC[0] == True:
                accuracy_tot = accuracy_tot + (1 == OC)

                accuracy_cnt+=1
                accuracy = float(accuracy_tot)/accuracy_cnt
            '''
            
            
            '''if ('%.2f' % prev_accuracy)  == ('%.2f' % accuracy):
                accuracy_same_cnt+=1
            else:
                accuracy_same_cnt = 0'''
                    

            '''if (accuracy_same_cnt >= 10) and (gradient_val < 1e-3):
                gradient_val = gradient_val#float(gradient_val*2)
                    accuracy_same_cnt = 0'''
            # prev_accuracy = accuracy

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
            # print("Actual Label: {} , NN Label: {},  Is Prediction Correct? {}, Error: {}, Cost Function: {}, Accuracy: {}, Gradient Value: {}".format('%.5f' % A_c, '%.5f' % NN_c, OC, '%.2E' % E, '%.2E' % CF, '%.5F' % accuracy, '%.2E' % gradient_val))
            print("Cost Function: ", CF)
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
            
            # Noitifies we are done processing
        #print("Done processing frames...")
            
            # Reports progress on number of images processed so far
        #print('Num Images: {} , Images Processed: {}'.format(num_images, images_processed))
        #images_processed+=1

    else:
        print("\nInvalid network")


