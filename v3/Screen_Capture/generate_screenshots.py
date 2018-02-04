# ================================ LOCAL IMPORTS ================================ #

import screen_capture as sc
import datetime
import time

# ================================ MAIN ================================ #

if __name__ == '__main__':

    rgb = sc.initialize_rgb_array()

    directory = r'C:\\Users\\Veda Sadhak\\Desktop\\raw_dataset'
    file_name = r'\\raw_'

    for i in range (0,500):

        initial_time = datetime.datetime.now()

        rgb = sc.get_screen_capture(rgb, directory=directory, filename=file_name+str(i), save=True, show_image=False)
        elapsed_time = datetime.datetime.now()-initial_time
        #print(rgb)
        print(elapsed_time)
        time.sleep(0.5)