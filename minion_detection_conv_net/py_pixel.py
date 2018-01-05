import pyscreenshot as pys
import numpy as np

#-----------------------

def get_image(x1=1920-1024,y1=232,x2=1920,y2=1000,padding=20):

    screenshot = pys.grab()
    region_of_interest = screenshot.crop((x1-padding,y1-padding,x2+padding,y2+padding))
    return region_of_interest

#-----------------------

def get_frames(image,frame_size,stride):

    rgb_image = image.convert()

    image_w , image_l = rgb_image.size

    num_frames_x = int(np.floor( (image_w - frame_size[0]) / stride[0] ) + 1)
    num_frames_y = int(np.floor( (image_l - frame_size[1]) / stride[1] ) + 1)

    frames = [[0 for y in range(num_frames_y)] for x in range(num_frames_x)]

    for x in range(0,num_frames_x):
        for y in range(0,num_frames_y):
            frame_x = x*frame_size[0]
            frame_y = y*frame_size[1]

            frames[x][y] = rgb_image.crop((frame_x, frame_y, frame_x + frame_size[0], frame_y + frame_size[1]))

    return frames, num_frames_x, num_frames_y

#-----------------------

def get_pixels(image):
    rgb_im = image.convert()
    x_max, y_max = rgb_im.size

    im_data = [[0 for y in range(y_max)] for x in range(x_max)]

    for x in range(0, x_max):
        for y in range(0, y_max):

            r,g,b = rgb_im.getpixel((x, y))

            r = r / 255.0
            g = g / 255.0
            b = b / 255.0

            im_data[x][y] = ([r, g, b])

    return im_data

#-----------------------

def get_image_size(image):

    rgb_im = image.convert()
    image_w,image_l = rgb_im.size
    image_h = 3

    return image_w,image_l,image_h