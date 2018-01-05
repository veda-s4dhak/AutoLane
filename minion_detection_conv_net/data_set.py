# ===================================== IMPORTS ===================================== #

import os
import numpy as np

# ================================== GLOBAL VARIABLES ================================== #

global processed_data_path
processed_data_path = r'C:\Users\Veda Sadhak\Google Drive\A Different Time\AI Projects\Autolane\processed_dataset'

global processed_image_tot
processed_image_tot = 0

# ================================== MAIN ================================== #

def get_num_processed_images():

    file_list = os.listdir(processed_data_path)

    global valid_file_list
    valid_file_list = []

    for file in file_list:
        if "edited" in file:
            valid_file_list.append(file)

    return len(valid_file_list)

#-----------------------

def get_processed_data_path():

    return processed_data_path

#-----------------------

def get_image(image_num):

    image_path = processed_data_path + r'\edited_{}'.format(image_num) + '.png'

    return image_path

#-----------------------

def get_image_data(image_num,image_dimensions,frame_size):

    # Getting bounding box data from text file
    bb_data_path = processed_data_path + r'\bounding_boxes_{}'.format(image_num) + '.txt'
    file = open(bb_data_path,'r')
    bounding_box_data = file.read()

    # Getting data from string format to list
    bounding_box_data = bounding_box_data[1:len(bounding_box_data)-1]
    bounding_box_data = bounding_box_data.replace('],', '.')
    bounding_box_data = bounding_box_data.replace(']', '')
    bounding_box_data = bounding_box_data.replace('[','')
    bounding_box_data = bounding_box_data.replace(' ','')
    if len(bounding_box_data) > 0:
        bounding_box_data = bounding_box_data.split('.')

    frame_x_tot = int(np.floor(image_dimensions[0] / frame_size[0]))
    frame_y_tot = int(np.floor(image_dimensions[1] / frame_size[1]))

    aggregated_img_data = [[ [-1.0,-1.0,-1.0,-1.0,0,0] for y in range(frame_y_tot)] for x in range(frame_x_tot)]

    for i in range(0,len(bounding_box_data)):

        if i % 2 == 0:
            # Getting x1,y1,x2,y2 of bounding box
            x1 = float(bounding_box_data[i].split(',')[0])
            y1 = float(bounding_box_data[i].split(',')[1])
            x2 = float(bounding_box_data[i+1].split(',')[0])
            y2 = float(bounding_box_data[i+1].split(',')[1])

            x1_relative = x1 % frame_size[0]
            y1_relative = y1 % frame_size[1]
            x2_relative = x2 % frame_size[0]
            y2_relative = y2 % frame_size[1]

            if x2_relative < x1_relative:
                x2_relative = frame_size[0]
            if y2_relative < y1_relative:
                y2_relative = frame_size[1]

            h = (y2_relative-y1_relative)/frame_size[0]
            w = (x2_relative-x1_relative)/frame_size[1]
            x = (((x2_relative - x1_relative) / 2) + x1_relative) / frame_size[0]
            y = (((y2_relative - y1_relative) / 2) + y1_relative) / frame_size[1]

            # Creating a coordinate list
            img_data = []
            img_data.append(h)
            img_data.append(w)
            img_data.append(x)
            img_data.append(y)

            # Generating and adding midpoints
            #coordinates.append(np.floor( ( (float(x2) - float(x1)) / 2) + float(x1) ))
            #coordinates.append(np.floor( ( (float(y2) - float(y1)) / 2) + float(y1) ))

            # Add class and probability
            img_data.append(1)
            img_data.append(1)

            frame_x = int(np.floor(float(x1) / frame_size[0]))
            frame_y = int(np.floor(float(y1) / frame_size[1]))

            aggregated_img_data[frame_x][frame_y] = img_data #Appending to final list

    return aggregated_img_data