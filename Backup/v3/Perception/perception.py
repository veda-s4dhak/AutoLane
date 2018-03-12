# ================================ GLOBAL IMPORT ================================ #

import sys
import pygame
import numpy as np

# ================================ Perception Screen Class ================================ #

class perception_screen():

    def __init__(self,screen_size_x,screen_size_y):

        pygame.init()

        # Initializing PyGame Window
        self.size = screen_size_x , screen_size_y
        self.screen = pygame.display.set_mode(self.size)

        self.red = (255,0,0)
        self.blue = (0,0,255)
        self.yellow = (255,255,0)
        self.green = (0,255,0)
        self.black = (0,0,0)

    def draw_rect(self,color,rect_x,rect_y,rect_w,rect_h):

        color_valid = 0

        if color == 'red':
            color_rgb = self.red
            color_valid = 1
        elif color == 'blue':
            color_rgb = self.blue
            color_valid = 1
        elif color == 'yellow':
            color_rgb = self.yellow
            color_valid = 1
        elif color == 'green':
            color_rgb = self.green
            color_valid = 1
        else:
            print("Invalid color specified please enter one of the following:")
            print("red,blue,yellow,green")

        if color_valid == 1:
            pygame.draw.rect(self.screen,color_rgb,(rect_x,rect_y,rect_w,rect_h))
            pygame.display.update()

            # Allowing user to move and close the pygame window
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

    def draw_matrix(self,matrix):

        # Initializing perception screen border offsets
        top_offset = 10
        right_offset = 10
        frame_y_offset = 5
        frame_x_offset = 5

        # Checking if Matrix Dimensions are Valid
        valid_matrix = 1
        matrix_dim = np.shape(matrix)
        if len(matrix_dim) != 2:
            valid_matrix = 0
            print("Error: Cannot update perception screen, invalid matrix -> matrix len: {}".format(len(matrix_dim)))

        if valid_matrix == 1:
            # Getting frame sizes
            frame_x_size = np.floor((self.size[0]-frame_x_offset)/matrix_dim[0])
            frame_y_size = np.floor((self.size[1]-frame_y_offset)/matrix_dim[1])

            self.screen.fill((0, 0, 0)) # Clearing screen

            # Drawing the frames
            for x in range(0,matrix_dim[0]):
                for y in range(0,matrix_dim[1]):
                    if matrix[x][y] == 1:
                        x1 = x*frame_x_size
                        y1 = y*frame_y_size
                        self.draw_rect('red',x1+right_offset,y1+top_offset,frame_x_size-2,frame_y_size-2)