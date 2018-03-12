# IMPORT
# ==================================================================================================
import sys
sys.path.insert(0, r'C:\Users\OM\Desktop\LOL-Autolane\LOL-Autolane')
from Settings import *


# MAIN
# ==================================================================================================

if __name__ == '__main__':

    screen_size_x = 341
    screen_size_y = 256

    x_pos = 10
    y_pos = 10
    width = 10
    height = 10

    x_dir = 1
    y_dir = 1

    ps = perception.perception_screen(screen_size_x,screen_size_y)

    # This block test the draw matrix method
    # ======================================
    while True:
        matrix = np.random.randint(0, 2, size=(25, 20))
        ps.draw_matrix(matrix)
        time.sleep(0.1)
    # ======================================

    # This block test the draw rect method
    # ======================================
    # color = "red"
    #
    # while True:
    #     ps.draw_
        # rect(color,x_pos,y_pos,width,height)
    #
    #     x_pos += 2*x_dir
    #     y_pos += 2*y_dir
    #
    #     if x_pos >= (screen_size_x - 10):
    #         x_dir = x_dir*-1
    #         color = "red"
    #     elif x_pos <= (10):
    #         x_dir = x_dir*-1
    #         color = "green"
    #
    #     if y_pos >= (screen_size_y - 10):
    #         y_dir = y_dir*-1
    #         color = "yellow"
    #     elif y_pos <= (10):
    #         y_dir = y_dir*-1
    #         color = "blue"
    # ======================================



