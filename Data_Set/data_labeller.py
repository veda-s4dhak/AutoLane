# IMPORT
# ==================================================================================================
import sys
sys.path.insert(0, r'C:\Users\OM\Desktop\LOL-Autolane\LOL-Autolane')
from Settings import *


# FUNCTIONS
# ==================================================================================================

# Number of processed images in processed_data_path
def get_num_processed_images():

    file_list = os.listdir(processed_dataset_dir)

    global valid_file_list
    valid_file_list = []

    for file in file_list:
        if "processed" in file:
            valid_file_list.append(file)

    return len(valid_file_list)


# Extracts midpoint data given processed image's image_num. Midpoints are returned
# in a list containing the midpoints in the form [x,y].
def get_image_midpoint_data(image_num):

    # Getting bounding box data from text file
    bb_data_path = processed_dataset_dir + r'\midpoints_{}'.format(image_num) + '.txt'
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
    
    data_list = []
    
    # Processes each midpoint data and appends the [x,y] pair to dataList
    for i in range(len(midpoint_data)):
        data = midpoint_data[i]
        split_data = data.split(', ')
        x = float(split_data[0])
        y = float(split_data[1])
        data_list.append([x,y])
    
    file.close()
    return(data_list)


# Fetches midpoint data for each image and returns data in a list. Each element of this list is
# a list containing midpoint data for some image. Each element of this list contains lists in form
# [x,y] specifying midpoint data.
def load_midpoint_data():
    num_images = get_num_processed_images()
    
    mid_point_data = []
    for i in range(0,num_images):
        mid_point_data.append(get_image_midpoint_data(i))
    return(mid_point_data)


# Assigns labels to each frame given midPoints, numPartsY and numPartsX.
def get_labels(mid_points, num_parts_y, num_parts_x, verbose = False):

    x_size = float(game_screen_x / num_parts_x)
    y_size = float(game_screen_y / num_parts_y)
    
    # Gets number of midpoints
    numMidPoints = len(mid_points)
    
    if(verbose):
        print('Number of MidPoints :', numMidPoints)
    # Initializes Used array for midpoints
    used = [False]*numMidPoints
    
    # Assigns labels of 1 or 0 based on if there's a midpoint inside this frame
    label_matrix = np.zeros((num_parts_y, num_parts_x), dtype=np.float32)
    for i in range(num_parts_x):
        for j in range(num_parts_y):
            x_min = float(i*x_size)
            y_min = float(j*y_size)
            x_max = float(x_min + x_size)
            y_max = float(y_min + y_size)
            
            if(verbose):
                print('Frame ', j, ' ', i, ' y_min: ', y_min, ' x_min: ', x_min, 'y_max: ', y_max, 'x_max: ', x_max)
            for k in range(len(used)):
                if (not used[k]) and (x_min <= float(mid_points[k][0]) < x_max) and (y_min <= float(mid_points[k][1]) < y_max):
                    used[k] = True
                    label_matrix[j][i] = 1

    return(label_matrix)


# Generates the labels matrices for all processed images.
def generate_label_matrix(mid_points, num_parts_y,
                          num_parts_x, save=True,verbose=False):

    # Gets number of processed images
    num_images = get_num_processed_images()
    
    label_matrix = np.zeros((num_images, num_parts_y, num_parts_x)).astype(np.float32)

    for i in range(num_images):
        current_label_matrix = get_labels(mid_points[i], num_parts_y, num_parts_x)
        label_matrix[i, :, :] = cp.deepcopy(current_label_matrix)

    if verbose == True:
        print("Image 1 Label Matrix")
        print(label_matrix[0])

    if save:
        np.savez(processed_dataset_dir + r'\label_matrix.npz', label_matrix)
        print("Created label matrix -> {}".format(processed_dataset_dir + r'\label_matrix.npz'))

    return(label_matrix)


# Loads the labels matrices for all processed images from an .npz file
def load_matrix():

    label_matrix = np.load(processed_dataset_dir + r'\label_matrix.npz')
    return label_matrix['arr_0']


# TEST CODE
# ==================================================================================================

if __name__ == '__main__':

    num_processed_images = get_num_processed_images()
    print("Num Processed Images: {}".format(num_processed_images))

    midpoint_data = load_midpoint_data()

    #for index in range(0,len(midpoint_data)):
    #    print("Image {} Midpoint Data: {}".format(index+1,midpoint_data[index]))

    #label_matrix = get_labels(1, midpoint_data[1], 341,256,20, 15)
    #print(label_matrix)

    generate_label_matrix(midpoint_data,20,20,True,False)

    time.sleep(3)

    label_matrix = load_matrix()
    #print(label_matrix)

