# IMPORT
# ==================================================================================================
import sys
sys.path.insert(0, r'C:\Users\OM\Desktop\LOL-Autolane\LOL-Autolane')
from Settings import *


# FUNCTIONS
# ==================================================================================================

# Initializes an array which has the same dimensions as the image being captured
def initialize_rgb_array():

    # Initializing RGB Array
    rgb = [[0 for x in range(game_screen_x)] for y in range(game_screen_y)]

    return rgb

# Captures the image on screen
def get_screen_capture(rgb,  # Contains the RGB array in which to store image pixel values
                       directory='',  # The directory in which to save the image
                       filename='',  # The filename of the directory to use when saving
                       save_image=False,  # If true then image is saved and vice versa
                       show_image=False):  # If true then image is show and vice versa

    # Getting screenshot
    windowname = 'This needs to be arbitrarily not empty.'
    hwnd = win32gui.FindWindow(None, windowname)
    wDC = win32gui.GetWindowDC(hwnd)
    dcObj = win32ui.CreateDCFromHandle(wDC)
    cDC = dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, game_screen_x,game_screen_y)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0, 0), (game_screen_x,game_screen_y), dcObj, (game_screen_x1, game_screen_y1),
               win32con.SRCCOPY)

    # Getting pixel values
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

        rgb[rgb_index // game_screen_x][rgb_index % game_screen_x] = ([r/255.0, g/255.0, b/255.0])  # ([r,g,b,a])

    # Generating image
    bmpStr = dataBitMap.GetBitmapBits(True)
    img = Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmpStr, 'raw', 'BGRX',
                           0, 1)

    # Showing image
    if show_image == True:
        img.show()

    # Saving file
    if save_image == True:
        dataBitMap.SaveBitmapFile(cDC, directory+filename+'.png')
        print('Saved image -> {}'.format(directory + filename + '.png'))

    # Freeing Memory
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())

    return rgb,img


# Gets the pixels of the image
def get_pixels(image):
    rgb_im = image.convert()
    x_max, y_max = rgb_im.size

    im_data = [[0 for x in range(x_max)] for y in range(y_max)]

    for x in range(0, x_max):
        for y in range(0, y_max):
            r, g, b = rgb_im.getpixel((x, y))

            r = r / 255.0
            g = g / 255.0
            b = b / 255.0

            im_data[y][x] = ([r, g, b])

    return im_data


# Gets the screen shots and saves them to a directory
def generate_screenshots(num_screen_shots=10,
                         time_interval=1,  # The time between screenshots
                         directory=raw_dataset_dir,
                         file_name=r'\\raw_'):

    rgb = sc.initialize_rgb_array()
    initial_time = datetime.datetime.now()

    for i in range (0,num_screen_shots):
        rgb = sc.get_screen_capture(rgb, directory=directory, filename=file_name+str(i),
                                    save=True, show_image=False)
        elapsed_time = datetime.datetime.now()-initial_time
        print(elapsed_time)
        time.sleep(time_interval)


# TEST CODE
# ==================================================================================================

if __name__ == '__main__':

    rgb = initialize_rgb_array()

    initial_time = datetime.datetime.now()

    for i in range(0,20):

        get_screen_capture(rgb, directory=r'C:\Users\OM\Desktop\Screen_Capture_Test',
                           filename='Test',save_image=False, show_image=True)
        elapsed_time = datetime.datetime.now() - initial_time
        print(elapsed_time)