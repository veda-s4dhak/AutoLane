import os
import pyscreenshot
from PIL import Image as PIL_image

# =======================================================================================================================

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

#=======================================================================================================================

def get_screenshots(num_of_screenshots, directory, file_name):
    for i in range(1, num_of_screenshots + 1):
        im = pyscreenshot.grab()
        im.save(directory + file_name + '{}.png'.format(i))

#=======================================================================================================================

def capture_sub_image(im, im_name, x_pixels, y_pixels, directory, size=75):
    if type(x_pixels) == int:
        x_pixels = [x_pixels]
        y_pixels = [y_pixels]

    for i in range(0,len(x_pixels)):
        new_directory = directory + r'\\Image_Set\\{}'.format(im_name)
        create_dir(new_directory)

        new_image = im.crop((x_pixels[i], y_pixels[i], x_pixels[i] + size, y_pixels[i] + size))
        #print("Saving new image: {}".format(new_directory + r'\\minion_{}.png'.format(i + 1)))
        new_image.save(new_directory + r'\\minion_{}.png'.format(i + 1),'PNG')

#=======================================================================================================================

def print_all_pixels(im):
    rgb_im = im.convert()
    x_max, y_max = rgb_im.size

    for x in range(1, x_max):
        for y in range(1, y_max):
            r,g,b,a = rgb_im.getpixel((x, y))
            print("X: {] Y: {} R: {} G: {} B: {} A: {}".format(x,y,r,g,b,a))

#=======================================================================================================================

def output_clusters_to_image(im, im_name, x_clusters, y_clusters, directory):
    pixel_map = im.load()

    color_index = 0; # 0 = change red, 1 = change green, 2 = change blue
    color = 255;

    for cluster_num in range(0, len(x_clusters)):

        for coord in range(0, len(x_clusters[cluster_num])):
            if (color_index == 0):
                pixel_map[x_clusters[cluster_num][coord], y_clusters[cluster_num][coord]] = (color, 255, 255, 255)
            if (color_index == 1):
                pixel_map[x_clusters[cluster_num][coord], y_clusters[cluster_num][coord]] = (255, color, 255, 255)
            if (color_index == 2):
                pixel_map[x_clusters[cluster_num][coord], y_clusters[cluster_num][coord]] = (255, 255, color, 255)

        # Changing color index for each cluster
        color_index += 1
        if (color_index >= 3):
            color_index = 0;

        # Changing color for each cluster
        color = color - 50
        if (color <= 0):
            color = 255;

    # Creating the new image
    x_max,y_max = im.size

    new_img = PIL_image.new(im.mode, im.size)
    new_pixel_map = new_img.load()

    for x in range(1, x_max):
        for y in range(1, y_max):
            new_pixel_map[x, y] = pixel_map[x, y]

    im.close()
    new_directory = directory + r'\\Image_Set\\{}'.format(im_name)
    create_dir(new_directory)
    new_img.save(new_directory + '\\cluster_image.png')
    new_img.close()

# =======================================================================================================================

def find_pixels_by_color(im,r_val=150,g_val=100,b_val=100):
    rgb_im = im.convert()
    x_max,y_max = rgb_im.size

    x_coordinate = []
    y_coordinate = []
    coordinate_cnt = 0

    for x in range(1, x_max):
        for y in range(1, y_max):
            r, g, b, a = rgb_im.getpixel((x, y))

            if (r > r_val) and (g < g_val) and (b < b_val):
                x_coordinate.append(x)
                y_coordinate.append(y)
                coordinate_cnt += 1

    return x_coordinate, y_coordinate, coordinate_cnt