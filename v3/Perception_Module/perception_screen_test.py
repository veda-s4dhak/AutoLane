# ================================ GLOBAL IMPORT ================================ #

import time

# ================================ LOCAL IMPORT ================================ #

import perception as p

# ================================ Main ================================ #

if __name__ == '__main__':

    screen_size_x = 1080
    screen_size_y = 940

    x_pos = 10
    y_pos = 10
    width = 10
    height = 10

    x_dir = 1
    y_dir = 1

    ps = p.perception_screen(screen_size_x,screen_size_y)

    while True:
        ps.draw_rect("red",x_pos,y_pos,width,height)

        x_pos += 2*x_dir
        y_pos += 2*y_dir

        if x_pos >= (screen_size_x - 10):
            x_dir = x_dir*-1
        elif x_pos <= (10):
            x_dir = x_dir*-1

        if y_pos >= (screen_size_y - 10):
            y_dir = y_dir*-1
        elif y_pos <= (10):
            y_dir = y_dir*-1



