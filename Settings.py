# GLOBAL IMPORTS
# =========================================

import sys
import os
import datetime
import time
import tensorflow as tf
from PIL import Image
import win_unicode_console
import win32gui
import win32ui
import win32con
import tkinter as tk
from PIL import Image,ImageTk
import numpy as np
import scipy.misc as smp
import threading
import pygame
import copy as cp


# GLOBALS
# =========================================

global real_time_flag
real_time_flag = False

global train_flag
train_flag = True

global test_flag
test_flag = False

global game_screen_x
game_screen_x = 400

global game_screen_y
game_screen_y = 300

global game_screen_x1
game_screen_x1 = 1015

global game_screen_y1
game_screen_y1 = 326

global game_screen_x2
game_screen_x2 = 1417

global game_screen_y2
game_screen_y2 = 654

global processed_dataset_dir
processed_dataset_dir = r'C:\Users\OM\Desktop\processed_dataset_{}x{}'.format(game_screen_y,
                                                                              game_screen_x)

global raw_dataset_dir
raw_dataset_dir = r'C:\Users\OM\Desktop\raw_dataset_{}x{}'.format(game_screen_y,
                                                                  game_screen_x)

global checkpoint_dir
checkpoint_dir = r'C:\Users\OM\Desktop\LOL-Autolane\LOL-Autolane\Checkpoint'\
                  .format(game_screen_x,game_screen_y)


# LOCAL IMPORTS
# =========================================

sys.path.insert(0, r'C:\Users\OM\Desktop\LOL-Autolane\LOL-Autolane\Screen_Capture')
import screen_capture as sc

sys.path.insert(0, r'C:\Users\OM\Desktop\LOL-Autolane\LOL-Autolane\Neural_Network')
import CNN

sys.path.insert(0, r'C:\Users\OM\Desktop\LOL-Autolane\LOL-Autolane\Perception')
import perception

sys.path.insert(0, r'C:\Users\OM\Desktop\LOL-Autolane\LOL-Autolane\Data_Set')
import data_generator as dg
import data_labeller as dl
import data_set as ds
