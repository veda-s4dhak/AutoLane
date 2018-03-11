# ===================================== IMPORTS ===================================== #

import tkinter as tk
from PIL import Image,ImageTk
import numpy as np
import scipy.misc as smp
import threading
import time

# ================================== GLOBAL VARIABLES ================================== #

global image_num
image_num = 300

global current_image

global base_dir
base_dir = r"C:\Users\OM\Desktop\raw_dataset"

global output_dir
output_dir = r"C:\Users\OM\Desktop\processed_dataset"

global pix
global raw_pix
global pix_ready_flag

pix_ready_flag = 1

global points_list
points_list = []

global midpoint_list
midpoint_list = []

# ================================== GUI ================================== #

def init_gui():

    # Getting starting image

    global current_image
    img_path = base_dir + r'\raw_300.png'
    current_image = r'\raw_300.png'
    rgb_image = Image.open(img_path).convert()
    image_w, image_l = rgb_image.size
    print(image_w,image_l)
    gp_thread = get_pixels_thread(1)
    gp_thread.start()

    # Initializing main gui window

    global gui
    gui = tk.Tk()
    gui.resizable(width=False,height=False)
    gui.geometry('{}x{}'.format(image_w+115,image_l+30))

    # Initializing canvas

    global canvas
    canvas = tk.Canvas(gui,width=image_w+20,height=image_l+20)
    canvas.place(x=10,y=10)

    global canvas_placeholder
    canvas_placeholder = canvas.create_image((10,10), image=(), anchor='nw')

    change_pic(img_path)

    # Initializing Buttons

    next_btn = tk.Button(gui,text="    Next   ",command=lambda:show_next_pic())
    next_btn.place(x=image_w+35,y=20)

    previous_btn = tk.Button(gui, text="Previous", command=lambda:show_previous_pic())
    previous_btn.place(x=image_w + 35, y=50)

    submit_btn = tk.Button(gui, text="  Submit ", command=lambda:submit_edited_pic())
    submit_btn.place(x=image_w + 35, y=80)

    clear_btn = tk.Button(gui, text="   Clear   ", command=lambda:print("Clear button pressed"))
    clear_btn.place(x=image_w + 35, y=110)

    canvas.bind("<Button 1>", mouse_click_event)

# ================================== BTN FCTNS ================================== #

def show_next_pic():

    global image_num
    global tk_img
    global current_image
    global points_list
    global midpoint_list

    if pix_ready_flag == 1:
        if image_num < 2000:
            image_num+=1
            img_path = base_dir + r'\raw_{}.png'.format(image_num)
            current_image = r'\raw_{}.png'.format(image_num)
            print(current_image)
            points_list = []
            midpoint_list = []
            change_pic(img_path)

            gp_thread = get_pixels_thread(1)
            gp_thread.start()
        else:
            print("Maximum image_count reached")
    else:
        print("Not ready getting pixels")

def show_previous_pic():

    global image_num
    global tk_img
    global current_image
    global points_list
    global midpoint_list

    if pix_ready_flag == 1:
        if image_num > 1:
            image_num-=1
            img_path = base_dir + r'\raw_{}.png'.format(image_num)
            current_image = r'\raw_{}.png'.format(image_num)
            print(current_image)
            points_list = []
            midpoint_list = []
            change_pic(img_path)

            gp_thread = get_pixels_thread(1)
            gp_thread.start()
        else:
            print("Minimum image count reached.")
    else:
        print("Not ready getting pixels")

def submit_edited_pic():

    pixels = np.fliplr(pix)
    pixels = np.rot90(pixels)

    new_img_path = output_dir + r'\processed_{}.png'.format(image_num)
    smp.imsave(new_img_path, pixels)

    file = open(output_dir+ r'\bounding_boxes_' + str(image_num) + '.txt', 'w+')
    file.seek(0)
    file.truncate()
    file.write(str(points_list))
    file.close()

    file = open(output_dir + r'\midpoints_' + str(image_num) + '.txt', 'w+')
    file.seek(0)
    file.truncate()
    file.write(str(midpoint_list))
    file.close()


# ================================== PIC Functions ================================== #

def change_pic(img_path):

    global canvas_placeholder
    global tk_img

    tk_img = ImageTk.PhotoImage(Image.open(img_path))
    canvas.itemconfigure(canvas_placeholder,image=tk_img)

    gui.update()

def mouse_click_event(event_origin):

    global x0,y0
    x0 = event_origin.x
    y0 = event_origin.y
    draw_point_on_image(x0,y0)

def draw_point_on_image(x_pos,y_pos):

    global pix

    while pix_ready_flag == 0:
         time.sleep(0.05)

    x_pos = x_pos-10
    y_pos = y_pos-10

    # Drawing the point

    pix[x_pos, y_pos] = [255, 255, 255]

    for i in range(1, 5):
        pix[x_pos + i, y_pos] = [255, 255, 255]
        pix[x_pos - i, y_pos] = [255, 255, 255]
        pix[x_pos, y_pos + i] = [255, 255, 255]
        pix[x_pos, y_pos - i] = [255, 255, 255]

    # Drawing box if valid

    num_points = update_point_states(x_pos,y_pos)

    if (num_points % 2) == 0:
        draw_box_flag = 1
    else:
        draw_box_flag = 0

    if draw_box_flag == 1:
        prev_x_pos = points_list[num_points - 2][0]
        prev_y_pos = points_list[num_points - 2][1]

        update_midpoint_state(prev_x_pos,prev_y_pos,x_pos,y_pos)

        for x in range(prev_x_pos,x_pos):
            pix[x , y_pos] = [255, 255, 255]
            pix[x , prev_y_pos] = [255, 255, 255]

        for y in range(prev_y_pos,y_pos):
            pix[x_pos , y] = [255, 255, 255]
            pix[prev_x_pos , y] = [255, 255, 255]

    pixels = np.fliplr(pix)
    pixels = np.rot90(pixels)

    new_img_path = output_dir + r'\temp.png'.format(image_num)
    smp.imsave(new_img_path, pixels)

    change_pic(new_img_path)

class get_pixels_thread(threading.Thread):

    def __init__(self, threadID):
        threading.Thread.__init__(self)
        self.threadID = threadID

    def run(self):

        global pix
        global raw_pix
        global pix_ready_flag

        pix_ready_flag = 0
        print('Getting pixels')

        rgb_image = Image.open(base_dir + current_image).convert()
        image_w, image_l = rgb_image.size
        pix = np.zeros((image_w, image_l, 3), dtype=np.uint8)

        for x in range(0, image_w):
            for y in range(0, image_l):
                r, g, b = rgb_image.getpixel((x, y))
                pix[x, y] = [r, g, b]

        raw_pix = np.fliplr(pix)
        raw_pix = np.rot90(raw_pix)
        raw_img_path = output_dir + r'\raw_{}.png'.format(image_num)
        smp.imsave(raw_img_path, raw_pix)
        smp.imsave(raw_img_path, raw_pix)

        print('Completed getting pixels')
        pix_ready_flag = 1

def update_point_states(x_pos,y_pos):

    global points_list
    points_list.append([x_pos,y_pos])
    print('pl->{}'.format(points_list))

    return len(points_list)

def update_midpoint_state(x1,y1,x2,y2):

    global midpoint_list

    midpoint_x = np.floor((x2 - x1) / 2) + x1
    midpoint_y = np.floor((y2 - y1) / 2) + y1

    midpoint_list.append([midpoint_x, midpoint_y])
    print('mpl->{}'.format(midpoint_list))

# ================================== MAIN ================================== #

if __name__ == '__main__':

    init_gui()

    gui.mainloop()