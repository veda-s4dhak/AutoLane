import os
import pyscreenshot
from PIL import Image as PIL_image
import time

# =======================================================================================================================

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

#=======================================================================================================================

def get_screenshots(num_of_screenshots, directory, file_name, x1=1920-1024,y1=232,x2=1920,y2=1000,padding=20):

    create_dir(directory)

    for i in range(1, num_of_screenshots + 1):
        im = pyscreenshot.grab()
        region_of_interest = im.crop((x1 - padding, y1 - padding, x2 + padding, y2 + padding))
        region_of_interest.save(directory + file_name + '{}.png'.format(i))
        print("Created screenshot: {}{}".format(filename,'{}.png'.format(i)))
        time.sleep(1)

if __name__ == '__main__':
    try:
        directory = r'C:\\Users\\HP_OWNER\\Desktop\\LOL-Autolane\\raw_dataset\\'
        filename = r'raw_'
        num_screenshots = 500
        get_screenshots(num_screenshots,directory,filename)
    except:
        print('Please press any key to close')
        input()