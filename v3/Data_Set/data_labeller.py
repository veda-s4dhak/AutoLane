'''

TODO

Clean up is basically change varable name and add comments

-> Load_Midpoint_Data (Already implemented - clean up) (Xiao Lei): Function to load midpoint data from text files
-> Generate_Label_Matrix (Xiao Lei): Function to generate label matrix from a single image (saves label matrix into a npz file)
-> Load_Output_Labels (Anish): Function to load the npz file containing output label matrixes
-> In the main function iterate through all the images and create label matrixes

'''

import os
import numpy as np
from PIL import Image
import copy
import random

global processed_data_path
processed_data_path = r'C:\\Users\\HP_OWNER\\Desktop\\LOL-Autolane\\processed_dataset\\'

# Specifies Width and Height of our images
global imgX
global imgY

# Divisors: 1 2 4 8 43 86 172 344
imgX = 344

# Divisors: 1 2 3 6 43 86 129 258
imgY = 258

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
Extracts midpoint data given processed image's image_num.
Midpoints are returned in a list containing the midpoints
in the form [x,y].

Example:
    dataList = [ [1,2], [2,3]]
    dataList[0] gives first midpoint of this image: [1,2]
    
Inputs:
    image_num: numbering of this processed image

Outputs:
    dataList: list of midpoints for this image
'''
def get_image_midpoint_data(image_num):

    # Getting bounding box data from text file
    bb_data_path = processed_data_path + r'\midpoints_{}'.format(image_num) + '.txt'
    file = open(bb_data_path,'r')
    midpoint_data = file.read()

    # Getting data from string format to list
    # Removes outer square bracket []
    midpoint_data = midpoint_data[1:-1]
    midpoint_data = midpoint_data.replace('], ', '#')
    midpoint_data = midpoint_data.replace(']', '')
    midpoint_data = midpoint_data.replace('[','')
    
    # Checks if we have any midpoints in midpoint_data
    if len(midpoint_data) > 0:
        midpoint_data =  midpoint_data.split('#')
    else:
        return([])
    
    dataList = []
    
    # Processes each midpoint data and appends the [x,y] pair to dataList
    for i in range(len(midpoint_data)):
        data = midpoint_data[i]
        splitData = data.split(', ')
        x = float(splitData[0])
        y = float(splitData[1])
        dataList.append([x,y])
    
    file.close()
    return(dataList)

'''
Fetches midpoint data for each image and returns
data in a list. Each element of this list is a list
containing midpoint data for some image. Each element 
of this list contains lists in form [x,y] specifying 
midpoint data.

Example:

    midPointData = [[[1,2],[3,4]], [[0,1],[2,3]]]
    midPoints[0][1] gives first midpoint for image 0: [1,2] 

Returns:
    midPointData: list of midpoint data for each image
'''
def Load_Midpoint_Data():
    numImages = get_num_processed_images()
    
    midPointData = []
    for i in range(numImages):
        midPointData.append(get_image_midpoint_data(i))
    return(midPointData)

'''
Assigns labels to each frame given midPoints, numPartsY,
and numPartsX.

Inputs:
        imageNum: numbering of this image in the series
        midPoints: np array of midpoints of dimension numMidpoints x 2
        numPartsX: number of frames to split X range
        numPartsY: number of frames to split Y range
        
Returns:
    Labels: a imgY x imgX dimension np array of 1's and 0's corresponding
    to the labelling of frames in image imageNum
'''
def getLabel(imageNum, midPoints, numPartsY, numPartsX, printEnabled = False):
    
    xSize = float(imgX // numPartsX)
    ySize = float(imgY // numPartsY)
    
    # Gets number of midpoints
    numMidPoints = len(midPoints)
    
    if(printEnabled):
        print('Number of MidPoints :', numMidPoints)
    # Initializes Used array for midpoints
    Used = [False]*numMidPoints
    
    # Assigns labels of 1 or 0 based on if there's a midpoint inside this frame
    Labels = np.zeros((numPartsY, numPartsX), dtype=np.float32)
    for i in range(numPartsX):
        for j in range(numPartsY):
            xMin = float(i*xSize)
            yMin = float(j*ySize)
            xMax = float(xMin + xSize)
            yMax = float(yMin + ySize)
            
            if(printEnabled):
                print('Frame ', j, ' ', i, ' yMin: ', yMin, ' xMin: ', xMin, 'yMax: ', yMax, 'xMax: ', xMax)
            for k in range(len(Used)):
                if (not Used[k]) and (xMin <= float(midPoints[k][0]) < xMax) and (yMin <= float(midPoints[k][1]) < yMax):
                    Used[k] = True
                    Labels[j][i] = 1.0
    return(Labels)
'''
Generates the labels matrices for all processed images.

Inputs:
    midPoints: np array of midpoints of dimension numMidpoints x 2
    numPartsX: number of frames to split X range
    numPartsY: number of frames to split Y range
    save: True iff the labels matrix is saved to labels.npz file
    
Returns:
    A numImages x imgY x imgX dimension np array of 1's and 0's corresponding
    to the labelling of the frames in each image in series
'''
def Generate_Label_Matrix(midPoints, numPartsY, numPartsX, save=True):
    
    # Gets number of processed images
    numImages = get_num_processed_images()
    
    labels = np.zeros((numImages, numPartsY, numPartsX)).astype(np.float32)
    for i in range(numImages):
        
        label = getLabel(i, midPoints[i], numPartsY, numPartsX)
        labels[i, :, :] = copy.deepcopy(label)
    
    if save:
        np.savez('labels.npz', labels = labels)
    return(labels)
    
    
            

