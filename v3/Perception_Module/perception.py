# ================================ GLOBAL IMPORT ================================ #

import sys
import pygame

# ================================ Perception Screen Class ================================ #

class perception_screen():

    def __init__(self,screen_size_x,screen_size_y):

        pygame.init()

        # Initializing PyGame Window
        size = width , height = screen_size_x , screen_size_y
        self.screen = pygame.display.set_mode(size)



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

