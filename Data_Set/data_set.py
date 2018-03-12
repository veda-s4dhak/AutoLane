# IMPORT
# ==================================================================================================
import sys
sys.path.insert(0, r'C:\Users\OM\Desktop\LOL-Autolane\LOL-Autolane')
from Settings import *


# FUNCTIONS
# ==================================================================================================

# Gets the number of processed images in processed_dataset_dir.
def get_num_processed_images():

    file_list = os.listdir(processed_dataset_dir)

    global valid_file_list
    valid_file_list = []

    for file in file_list:
        if "processed" in file:
            valid_file_list.append(file)

    return len(valid_file_list)


# Gets the data path of a particular image in processed_dataset_dir
def get_image_path(image_num):
    return(processed_dataset_dir + r'\raw_{}'.format(image_num) + '.png')


# Shuffles two np arrays along axis 0 in the same order.
def shuffle_two_np_arrays(l1, l2):
    numData = np.arange(l1.shape[0])
    np.random.shuffle(numData)
    return l1[numData], l2[numData]


# Returns a np array (type float32) containing all of the images in processed_dataset_dir
def load_images_from_dataset():
    
    num_images_total = get_num_processed_images()
    img_list = []

    for i in range(0,num_images_total):
        image = Image.open(get_image_path(i))
        pixels = sc.get_pixels(image)

        img_list.append(pixels)

    return(np.array(img_list).astype(np.float32))


# Splits imagesArray into images of type training, cross validation, and testing given imagesArray,
# nTrain, nValid, nTest
def split_data_set(imagesArray, nTrain, nValid, nTest):

    # nTrain = number of training images
    # nValid = number of cross
    # validation images and nTest = number of test images
    split_info = [nTrain, nValid, nTest]
    split_info_np = np.array(split_info).astype(np.int32)
    
    # Computes split indices using cumulative sum
    cumulative_split_info = np.cumsum(split_info_np, axis=0, dtype=np.int32)

    # Splitting frames
    split_frames = np.split(imagesArray, cumulative_split_info, 0)

    return(split_frames)


# Prepares data set given imgArray and labelsArray
def prepare_data_set(img_array, labels_array, num_train, num_valid, num_test):
    
    img_array_new, labels_array_new = shuffle_two_np_arrays(img_array, labels_array)
    
    split_img = split_data_set(img_array_new, num_train, num_valid, num_test)
    split_labels = split_data_set(labels_array_new, num_train, num_valid, num_test)

    # split_img: splitted image arrays in an np array
    # split_labels: splitted labels arrays in an np array
    return split_img, split_labels


# TEST CODE
# ==================================================================================================

if __name__ == '__main__':

    num_processed_images = get_num_processed_images()
    print('Num Processed Images: {}'.format(num_processed_images))

    images = load_images_from_dataset()
    print("Num Images: {}".format(len(images)))
    print("Image Width: {}".format(len(images[0])))
    print("Image Height: {}".format(len(images[0][0])))

    split_frames = split_data_set(images,10,5,5)
    print("Num of Frames: ".format(len(split_frames)))
    print("Num of Training Frames: {}".format(len(split_frames[0])))
    print("Num of Validation Frames: {}".format(len(split_frames[1])))
    print("Num of Test Frames: {}".format(len(split_frames[2])))
    print("Check: {}".format(len(split_frames[3])))
