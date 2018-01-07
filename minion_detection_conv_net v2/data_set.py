# ===================================== IMPORTS ===================================== #

import os
import numpy as np
from PIL import Image
import py_pixel as p
import copy
import random


# ================================== GLOBAL VARIABLES ================================== #

global processed_data_path
processed_data_path = r'C:\\Users\\HP_OWNER\\Desktop\\LOL-Autolane\\processed_dataset\\'

global processed_image_tot
processed_image_tot = 0

global numImagesTotal

# Frame settings:
# Specifies the neural network architecture: from line 31 to 60
global frame_size 
frame_size = [128, 128]
    
# Stride size (not overlapping) shift by 128
global frame_stride_size 
frame_stride_size = [128,128]

# Image data

# Images
global Images
Images = list()

# Image dimension: lxwxh
global fullImageDim
fullImageDim = list()

# Image bounding box data for each image in list
global imageBoundData
imageBoundData = list()

# Frame data for each image: [frames, num_frames_x, num_frames_y]
global frameData
frameData = list()

# Img data set of lists containing all data above
# Structure of each element in list: [Images[i], fullImageDim[i], imageBoundData[i], frameData[i]]
global imgDataSet
imgDataSet = list()


# ================================== MAIN ================================== #

def get_num_processed_images():

    file_list = os.listdir(processed_data_path)

    global valid_file_list
    valid_file_list = []

    for file in file_list:
        if "edited" in file:
            valid_file_list.append(file)

    return len(valid_file_list)

numImagesTotal = get_num_processed_images()
#-----------------------

def get_processed_data_path():

    return processed_data_path

#-----------------------

def get_image(image_num):

    image_path = processed_data_path + r'\edited_{}'.format(image_num) + '.png'

    return image_path

#----------------------
'''
Gets batch data given batch_size

batch_size: size of batch

'''
def get_image_batch(batch_size = 1):
    
    # Gets cloned batch of size batch_size
    return(list(random.sample(imgDataSet, batch_size)))
    
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

'''
   Initializes data for all processed image data
'''
def init_image_data():
    
    # Loops over each processed image
    for i in range(1, numImagesTotal+1):
        
        # Gets image
        image = Image.open(get_image(i))
        
        # Gets dimension of image
        full_image_w, full_image_l, full_image_h = p.get_image_size(image)
        
        # Gets bounding box data
        img_data = get_image_data(i, [full_image_w, full_image_l, full_image_h], frame_size)
            
        # Separates image into frames; separates image into grid
        frames, num_frames_x, num_frames_y = p.get_frames(image, frame_size, frame_stride_size)
        
        # appends data to lists
        Images.append(image)
        fullImageDim.append([full_image_w, full_image_l, full_image_h])
        imageBoundData.append(img_data)
        frameData.append([frames, num_frames_x, num_frames_y])
        
        # Append image data for current image to imageSet
        imgDataSet.append([Images[i-1], fullImageDim[i-1], imageBoundData[i-1], frameData[i-1]])
        
        
init_image_data()

# Test below
#cool = get_image_batch(batch_size=2)
#print(len(cool))
        
        