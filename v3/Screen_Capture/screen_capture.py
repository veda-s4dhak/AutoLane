# ================================ GLOBAL IMPORT ================================ #

import win32gui
import win32ui
import win32con
import time
import datetime
from PIL import Image

# ================================ FUNCTIONS ================================ #

def initialize_rgb_array():

    # Initializing Screenshot Parameters
    x1 = 1091
    y1 = 397
    x2 = 1435
    y2 = 655
    image_width = int(x2 - x1)
    image_height = int(y2 - y1)

    # Initializing RGB Array
    rgb = [[0 for y in range(image_height)] for x in range(image_width)]

    return rgb

def get_screen_capture(rgb,directory='',filename='',save=False,show_image=False):

    # Initializing Screenshot Parameters
    x1 = 1091
    y1 = 397
    x2 = 1435
    y2 = 655
    image_width = int(x2 - x1)
    image_height = int(y2 - y1)

    # Getting screenshot
    windowname = 'This needs to be arbitrarily not empty.'
    hwnd = win32gui.FindWindow(None, windowname)
    wDC = win32gui.GetWindowDC(hwnd)
    dcObj = win32ui.CreateDCFromHandle(wDC)
    cDC = dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, image_width, image_height)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0, 0), (image_width, image_height), dcObj, (x1, y1), win32con.SRCCOPY)

    # Getting Pixel Values
    bmpinfo = dataBitMap.GetInfo()
    bmpInt = dataBitMap.GetBitmapBits(False)
    num_pixels = int(len(bmpInt)) / 4
    for rgb_index in range(0, int(num_pixels)):
        r = bmpInt[rgb_index * 4+2]
        if r < 0: r = 256 + r
        g = bmpInt[rgb_index * 4+1]
        if g < 0: g = 256 + g
        b = bmpInt[rgb_index * 4]
        if b < 0: b = 256 + b

        rgb[rgb_index % image_width][rgb_index // image_width] = ([r,g,b])  # ([r,g,b,a])

    # Showing image
    if show_image == True:
        bmpStr = dataBitMap.GetBitmapBits(True)
        im = Image.frombuffer('RGB',(bmpinfo['bmWidth'],bmpinfo['bmHeight']),bmpStr,'raw','BGRX',0,1)
        im.show()

    # Saving file
    if save == True:
        print('Created file: {}'.format(directory+filename+'.png'))
        dataBitMap.SaveBitmapFile(cDC, directory+filename+'.png')

    # Freeing Memory
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())

    return rgb

def get_pixels(image):
    rgb_im = image.convert()
    x_max, y_max = rgb_im.size

    im_data = [[0 for y in range(y_max)] for x in range(x_max)]

    for x in range(0, x_max):
        for y in range(0, y_max):
            r, g, b = rgb_im.getpixel((x, y))

            r = r
            g = g
            b = b

            im_data[x][y] = ([r, g, b])

    return im_data

# ================================ TEST CODE ================================ #


    # if __name__ == '__main__':
#
#     rgb = initialize_rgb_array()
#
#     initial_time = datetime.datetime.now()
#
#     for i in range(1,20):
#
#         get_screen_capture(rgb, filename='Test',save=True, show_image=True)
#         input()
#         elapsed_time = datetime.datetime.now() - initial_time
#         print(elapsed_time)