# ================================ LOCAL IMPORTS ================================ #

import screen_capture as sc
import time

# ================================ MAIN ================================ #

if __name__ == '__main__':

    rgb = sc.initialize_rgb_array()

    directory = r'C:\\Users\\Veda Sadhak\\Desktop\\LOL-Autolane\\raw_dataset'
    file_name = r'\\raw_'

    for i in range (0,10):

        sc.get_screen_capture(rgb, directory=directory, filename=file_name+str(i), save=True, show_image=False)
        time.sleep(0.5)