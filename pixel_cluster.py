import os
import pyscreenshot
from PIL import Image as PIL_image

#=======================================================================================================================

def find_clusters(x_coordinates, y_coordinates, x_cutoff_dis=40, y_cutoff_dis=20):
    x_cluster_list = []
    y_cluster_list = []

    while ( len(x_coordinates) > 0 ): #while there are still pixels remaining that have not been assigned to a cluster

        # =========Initializing a new cluster========= #
        x_cluster = []
        y_cluster = []
        x_cluster.append(x_coordinates[0])
        y_cluster.append(y_coordinates[0])
        del x_coordinates[0]
        del y_coordinates[0]

        # =========Creating a cluster========= #
        i = 0
        while ( i < len(x_coordinates) ): # iterating through all of the pixels
            for j in range(0,len(x_cluster)):

                if (i >= len(x_coordinates)): # while condition is not being checked within for loop so this is required
                    break

                if ( is_pixel_adjacent(x_coordinates[i],y_coordinates[i],x_cluster[j],y_cluster[j],x_cutoff_dis,y_cutoff_dis) == True ):
                    x_cluster.append(x_coordinates[i])
                    y_cluster.append(y_coordinates[i])
                    del x_coordinates[i]
                    del y_coordinates[i]
                    i = 0
                else:
                    i += 1

        x_cluster_list.append(x_cluster)
        y_cluster_list.append(y_cluster)

    return x_cluster_list , y_cluster_list

#========================================================================================================================

def filter_cluster_by_size(x_clusters, y_clusters, size=50):
    for cluster_num in range(0,len(x_clusters)):

        if (cluster_num >= len(x_clusters)): #clusters are being dynamically deleted so this is required
            break

        if len(x_clusters[cluster_num]) < size:
            del x_clusters[cluster_num]
            del y_clusters[cluster_num]

    return x_clusters, y_clusters;

#========================================================================================================================

def is_pixel_adjacent(x1, y1, x2, y2, x_cutoff_dis, y_cutoff_dis):
    adjacent = False

    if ( (abs(x1 - x2) <= x_cutoff_dis) and (abs(y1 - y2) <= y_cutoff_dis) ):
        adjacent = True

    return adjacent

#========================================================================================================================

def find_upper_left_pixel(x_cluster, y_cluster):
    lowest_x = 9999
    lowest_y = 9999

    for i in range(0,len(y_cluster)):
        if y_cluster[i] < lowest_y:
            lowest_y = y_cluster[i]

    for i in range(0,len(x_cluster)):
        if y_cluster[i] == lowest_y:
            if x_cluster[i] < lowest_x:
                lowest_x = x_cluster[i]

    return lowest_x , lowest_y
