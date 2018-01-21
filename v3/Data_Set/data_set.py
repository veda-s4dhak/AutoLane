'''

TODO

Clean up is basically change varable name and add comments

-> Load_Images (Already Implemented - clean up) (Xiao Lei): Function to load images (returns images in an array)


-> Generate_Pixels (Already Implement - clean up) (Anish): Function to generate pixels given an image
-> Split_Data_Set (Already Implement - clean up) (Xiao Lei: Function to split image set (returns training and testing arrays)

'''
import os
import numpy as np
from PIL import Image
import py_pixel as p
import copy
import random

global processed_data_path
processed_data_path = r'C:\\Users\\HP_OWNER\\Desktop\\LOL-Autolane\\processed_dataset\\'

'''
Returns: Number of processed images in processed_data_path
'''
def get_num_processed_images():

    file_list = os.listdir(processed_data_path)

    global valid_file_list
    valid_file_list = []

    for file in file_list:
        if "processed" in file:
            valid_file_list.append(file)

    return len(valid_file_list)


'''
Input:
image_num: the number of processed images in processed_data_path

Returns:
Path of image image_num
'''
def get_image_path(image_num):
    return(processed_data_path + r'\processed_{}'.format(image_num) + '.png')

'''
Returns:
np array of type float32 containing all the images loaded
'''
def Load_Images():
    
    numImagesTotal = get_num_processed_images()
    imgList = []
    for i in range(numImagesTotal):
        
        # Gets image
        image = Image.open(get_image_path(i))
        pixels = p.get_pixels(image)
        imgList.append(pixels)
    return(np.array(imgList).astype(np.float32))

'''
Splits imagesArray into images of type training, cross validation, and testing
given imaagesArray, nTrain, nValid, nTest

Inputs: 
imagesArray: numData x l x w x 3 np array of image data of type np.float32
nTrain: number of training images
nValid: number of cross validation images
nTest: number of test images

Outputs:
Returns resulting training, validation, and testing images array in an np array
of type float32
'''
def Split_Data_Set(imagesArray, nTrain, nValid, nTest):
    
    splitInfo = [nTrain, nValid, nTest]
    splitInfoNP = np.array(splitInfo).astype(np.int32)
    
    # Computes split indices using cumulative sum
    cumSplitInfo = np.cumsum(splitInfoNP, axis=0, dtype=np.int32)
    
    splitFrames = np.split(imagesArray, cumSplitInfo, 0)
    return(splitFrames)
    
    
    