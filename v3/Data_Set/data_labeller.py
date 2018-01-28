import os
import numpy as np
import copy

global processed_data_path
processed_data_path = r'C:\\Users\\HP_OWNER\\Desktop\\LOL-Autolane\\processed_dataset'

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

def load_midpoint_data():
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

def get_labels(imageNum, midPoints, imgX, imgY, numPartsY, numPartsX, verbose = False):
    
    xSize = float(imgX // numPartsX)
    ySize = float(imgY // numPartsY)
    
    # Gets number of midpoints
    numMidPoints = len(midPoints)
    
    if(verbose):
        print('Number of MidPoints :', numMidPoints)
    # Initializes Used array for midpoints
    Used = [False]*numMidPoints
    
    # Assigns labels of 1 or 0 based on if there's a midpoint inside this frame
    label_matrix = np.zeros((numPartsY, numPartsX), dtype=np.float32)
    for i in range(numPartsX):
        for j in range(numPartsY):
            xMin = float(i*xSize)
            yMin = float(j*ySize)
            xMax = float(xMin + xSize)
            yMax = float(yMin + ySize)
            
            if(verbose):
                print('Frame ', j, ' ', i, ' yMin: ', yMin, ' xMin: ', xMin, 'yMax: ', yMax, 'xMax: ', xMax)
            for k in range(len(Used)):
                if (not Used[k]) and (xMin <= float(midPoints[k][0]) < xMax) and (yMin <= float(midPoints[k][1]) < yMax):
                    Used[k] = True
                    label_matrix[j][i] = 1
    return(label_matrix)

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

def generate_label_matrix(midPoints, imgX, imgY, numPartsY, numPartsX, save=True,verbose=False):
    
    # Gets number of processed images
    numImages = get_num_processed_images()
    
    label_matrix = np.zeros((numImages, numPartsY, numPartsX)).astype(np.float32)

    for i in range(numImages):
        current_label_matrix = get_labels(i, midPoints[i], imgX, imgY, numPartsY, numPartsX)
        label_matrix[i, :, :] = copy.deepcopy(current_label_matrix)

    if verbose == True:
        print("Image 1 Label Matrix")
        print(label_matrix[0])

    if save:
        np.savez(processed_data_path + r'\label_matrix.npz', label_matrix)

    return(label_matrix)

'''
Loads the labels matrices for all processed images from an .npz file

Inputs:
    image_num: this is the number of the image of which you which to get the label matrix

Returns:
    label_matrix: matrix containing the labels
'''

def load_matrix():

    label_matrix = np.load(processed_data_path + r'\label_matrix.npz')
    return label_matrix['arr_0']

# ==================================================== TEST CODE ==================================================== #



'''if __name__ == '__main__':

    num_processed_images = get_num_processed_images()
    print("Num Processed Images: {}".format(num_processed_images))

    midpoint_data = load_midpoint_data()

    #for index in range(0,len(midpoint_data)):
    #    print("Image {} Midpoint Data: {}".format(index+1,midpoint_data[index]))

    #label_matrix = get_labels(1, midpoint_data[1], 341,256,20, 15)
    #print(label_matrix)

    generate_label_matrix(midpoint_data,344,258,6,8,True,False)

    time.sleep(3)

    label_matrix = load_matrix()
    print(label_matrix)'''

