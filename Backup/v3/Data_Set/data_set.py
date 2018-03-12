import os
import numpy as np
from PIL import Image
import data_labeller as dlb

global processed_data_path
processed_data_path = r"C:\Users\OM\Desktop\processed_dataset"

'''
Returns:
    Number of processed images in processed_data_path
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
Shuffles two np arrays along axis 0 in the same order.

Inputs:
    l1, l2: np arrays to be shuffled in unison

Output: 
    arrays shuffled in the same order
'''

def shuffleTwoNPArrays(l1, l2):
    numData = np.arange(l1.shape[0])
    np.random.shuffle(numData)
    return l1[numData], l2[numData]

'''
Returns:
    np array of type float32 containing all the images loaded
'''

def load_images():
    
    numImagesTotal = get_num_processed_images()
    imgList = []
    for i in range(0,numImagesTotal):
        
        # Gets image
        image = Image.open(get_image_path(i))
        pixels = get_pixels(image)
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

Returns:
    Resulting training, validation, and testing images array in an np array
    of type float32
'''

def split_data_set(imagesArray, nTrain, nValid, nTest):
    
    splitInfo = [nTrain, nValid, nTest]
    splitInfoNP = np.array(splitInfo).astype(np.int32)
    
    # Computes split indices using cumulative sum
    cumSplitInfo = np.cumsum(splitInfoNP, axis=0, dtype=np.int32)
    
    splitFrames = np.split(imagesArray, cumSplitInfo, 0)
    return(splitFrames)
    
'''
Returns RGB pixels given an image

Inputs: 
    image: image which is being processed

Returns:
    rgb_data: return matrix containing RGB values corresponding to image
    
'''

def get_pixels(image):

    rgb_im = image.convert()
    x_max, y_max = rgb_im.size

    rgb_data = [[0 for x in range(x_max)] for y in range(y_max)]

    for x in range(0, x_max):
        for y in range(0, y_max):
            r, g, b = rgb_im.getpixel((x, y))

            r = r/255.0
            g = g/255.0
            b = b/255.0

            rgb_data[y][x] = ([r, g, b])

    return rgb_data
'''
Prepares data set given imgArray and labelsArray
Inputs:
    imgArray
    labelsArray
    nTrain
    nValid
    nTest
Returns:
    Tuple containing:
    splitImg: splitted image arrays in an np array
    splitLabels: splitted labels arrays in an np array
'''
def prepareDataSet(imgArray, labelsArray, nTrain, nValid, nTest):
    
    imgArrayNew, labelsArrayNew = shuffleTwoNPArrays(imgArray, labelsArray)
    
    splitImg = split_data_set(imgArrayNew, nTrain, nValid, nTest)
    splitLabels = split_data_set(labelsArrayNew, nTrain, nValid, nTest)
    
    return splitImg, splitLabels
    
# ==================================================== TEST CODE ==================================================== #

'''

if __name__ == '__main__':

    num_processed_images = get_num_processed_images()
    print('Num Processed Images: {}'.format(num_processed_images))

    images = load_images()
    print("Num Images: {}".format(len(images)))
    print("Image Width: {}".format(len(images[0])))
    print("Image Height: {}".format(len(images[0][0])))

    split_frames = split_data_set(images,10,5,5)
    print("Num of Frames: ".format(len(split_frames)))
    print("Num of Training Frames: {}".format(len(split_frames[0])))
    print("Num of Validation Frames: {}".format(len(split_frames[1])))
    print("Num of Test Frames: {}".format(len(split_frames[2])))
    print("Check: {}".format(len(split_frames[3])))

'''