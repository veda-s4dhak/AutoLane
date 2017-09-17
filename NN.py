import os
import pyscreenshot
import pixel_cluster
import scr_shot
import numpy
import random
from PIL import Image as PIL_image

#========================================================================================================================

def get_image_pixels(im):
    rgb_im = im.convert()
    x_max, y_max = rgb_im.size

    for x in range(1, x_max):
        for y in range(1, y_max):
            r,g,b,a = rgb_im.getpixel((x, y))
            print("X: {} Y: {} R: {} G: {} B: {} A: {}".format(x,y,r,g,b,a))

#=======================================================================================================================

def get_feature(feature_number_x,feature_number_y,im):
    rgb_im = im.convert()

    x_init = feature_number_x*5
    y_init = feature_number_x*5

    #print("{} | {} | {} | {} | {}".format(feature_number,x_init,x_init+5,y_init,y_init+5))

    feature_r = numpy.zeros((5,5))
    feature_g = numpy.zeros((5,5))
    feature_b = numpy.zeros((5,5))

    y_cnt = 0

    for y in range(y_init,y_init+5):
        x_cnt = 0

        for x in range(x_init,x_init+5):
            r, g, b, a = rgb_im.getpixel((x, y))
            feature_r[x_cnt][y_cnt] = r/255
            feature_g[x_cnt][y_cnt] = g/255
            feature_b[x_cnt][y_cnt] = b/255
            x_cnt += 1

        y_cnt += 1

    return feature_r , feature_g , feature_b

#=======================================================================================================================

def initialize_feature_comparators():
    feature_comparator_r = []

    for i in range (0,10):
        feature_comparator_r.append( numpy.zeros((5,5)) )

        for x in range(0,5):
            for y in range(0,5):
                feature_comparator_r[i][x][y] = random.randint(-100,100) / 100

    feature_comparator_g = []

    for i in range(0, 10):
        feature_comparator_g.append(numpy.zeros((5, 5)))

        for x in range(0, 5):
            for y in range(0, 5):
                feature_comparator_g[i][x][y] = random.randint(-100, 100) / 100

    feature_comparator_b = []

    for i in range(0, 10):
        feature_comparator_b.append(numpy.zeros((5, 5)))

        for x in range(0, 5):
            for y in range(0, 5):
                feature_comparator_b[i][x][y] = random.randint(-100, 100) / 100

    return feature_comparator_r , feature_comparator_g , feature_comparator_b

#=======================================================================================================================

def compute_feature_maps(feature_r,feature_g,feature_b,feature_comparator_r,feature_comparator_g,feature_comparator_b):
    feature_map_r = []
    feature_map_g = []
    feature_map_b = []

    for i in range(0,len(feature_comparator_r)):
        feature_map_r.append( numpy.multiply(feature_r , feature_comparator_r[i]) )
        feature_map_g.append( numpy.multiply(feature_g , feature_comparator_g[i]) )
        feature_map_b.append( numpy.multiply(feature_b , feature_comparator_b[i]) )

    return feature_map_r , feature_map_g , feature_map_b

# ======================================================================================================================

def compute_ReLU( feature_map_r,feature_map_g,feature_map_b ):

    for i in range(0,len(feature_map_r)):
        for x in range(0,5):
            for y in range(0,5):
                if (feature_map_r[i][x][y] < 0):
                    feature_map_r[i][x][y] = 0

                if (feature_map_g[i][x][y] < 0):
                    feature_map_g[i][x][y] = 0

                if (feature_map_b[i][x][y] < 0):
                    feature_map_b[i][x][y] = 0

    return feature_map_r , feature_map_g , feature_map_b

# ======================================================================================================================

def compute_max_pool( feature_map_r,feature_map_g,feature_map_b ):
    largest_r_val = []
    largest_g_val = []
    largest_b_val = []

    for i in range(0,len(feature_map_r)):

        largest_r_val.append(-999)
        largest_g_val.append(-999)
        largest_b_val.append(-999)

        for x in range(0,5):
            for y in range(0,5):
                if (feature_map_r[i][x][y] > largest_r_val[i]):
                    largest_r_val[i] = feature_map_r[i][x][y]

                if (feature_map_g[i][x][y] > largest_g_val[i]):
                    largest_g_val[i] = feature_map_g[i][x][y]

                if (feature_map_b[i][x][y] > largest_b_val[i]):
                    largest_b_val[i] = feature_map_b[i][x][y]

    return largest_r_val , largest_g_val , largest_b_val

# ======================================================================================================================

if __name__ == '__main__':

    directory = r'C:\\Users\\Veda Sadhak\\Desktop\\A Different Time\\AI Projects\\Minion_Detection\\'

    im = PIL_image.open(directory + r'test_image.png')

    feature_r , feature_g , feature_b = get_feature(0,0,im)

    # print("------------------")
    # print(feature_r)
    # print("------------------")
    # print(feature_g)
    # print("------------------")
    # print(feature_b)
    # print("------------------")

    feature_comparator_r,feature_comparator_g,feature_comparator_b = initialize_feature_comparators()

    # print("------------------")
    # print(feature_comparator_r[0])
    # print("------------------")
    # print(feature_comparator_g[0])
    # print("------------------")
    # print(feature_comparator_b[0])
    # print("------------------")

    feature_map_r , feature_map_g, feature_map_b = compute_feature_maps( feature_r,feature_g,feature_b,feature_comparator_r,feature_comparator_g,feature_comparator_b )

    # print("------------------")
    # print(feature_map_r[0])
    # print("------------------")
    # print(feature_map_g[0])
    # print("------------------")
    # print(feature_map_b[0])
    # print("------------------")

    feature_map_r, feature_map_g, feature_map_b = compute_ReLU(feature_map_r, feature_map_g, feature_map_b)

    print("------------------")
    print(feature_map_r[0])
    print("------------------")
    print(feature_map_g[0])
    print("------------------")
    print(feature_map_b[0])
    print("------------------")

    lr,lg,lb = compute_max_pool( feature_map_r,feature_map_g,feature_map_b )

    print(lr)