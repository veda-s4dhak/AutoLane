import os
import pyscreenshot
import pixel_cluster
import scr_shot
from PIL import Image as PIL_image

# =======================================================================================================================

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# =======================================================================================================================

if __name__ == '__main__':
    # get_screenshots(5,r'C:\\Users\\Veda Sadhak\\Google Drive\\A Different Time\\AI Projects\\Minion_Detection\\Image_Set\\',r'league')

    directory = r'C:\\Users\\Veda Sadhak\\Desktop\\A Different Time\\AI Projects\\Minion_Detection\\'
    r_val = 150
    g_val = 100
    b_val = 100

    create_dir(directory)

    for im_number in range (100,110):

        im = PIL_image.open(directory + r'Image_Set\\league{}.png'.format(im_number))
        print( "Opening image: {}".format(directory + r'Image_Set\\league{}.png'.format(im_number)) )
        im_name = r'league{}'.format(im_number)

        print("Finding Red Pixels...")

        r_pix_x_coord, r_pix_y_coord, coord_cnt = scr_shot.find_pixels_by_color(im, r_val, g_val, b_val)

        print("Red Pixel X Coordinates: {}".format(r_pix_x_coord))
        print("Red Pixel Y Coordinates: {}".format(r_pix_y_coord))
        print("Coordinate Count: {}".format(coord_cnt))

        print("Done Finding Red Pixels...")
        print()

        # ========================================================================

        print("Finding Clusters...")

        x_clusters, y_clusters = pixel_cluster.find_clusters(r_pix_x_coord, r_pix_y_coord, 40, 20)

        print("X Clusters: {}".format(x_clusters))
        print("Size X Cluster: {}".format(len(x_clusters)))
        print("Y Clusters: {}".format(y_clusters))
        print("Size Y Cluster: {}".format(len(y_clusters)))

        print("Done Finding Clusters...")
        print()

        # ========================================================================

        print("Filtering Clusters...")

        fil_x_clusters, fil_y_clusters = pixel_cluster.filter_cluster_by_size(x_clusters, y_clusters, 50)

        print("Filtered X Clusters: {}".format(fil_x_clusters))
        print("Size Filtered X Cluster: {}".format(len(fil_x_clusters)))
        print("Filtered Y Clusters: {}".format(fil_y_clusters))
        print("Size Filtered Y Cluster: {}".format(len(fil_y_clusters)))

        print("Done Filtering Clusters...")
        print()

        # ========================================================================

        print("Finding upper left pixel of each cluster...")

        upper_left_pix_x_list = []
        upper_left_pix_y_list = []

        for cluster_num in range(0, len(fil_x_clusters)):
            upper_left_pix_x, upper_left_pix_y = pixel_cluster.find_upper_left_pixel(fil_x_clusters[cluster_num], fil_y_clusters[cluster_num])
            upper_left_pix_x_list.append(upper_left_pix_x)
            upper_left_pix_y_list.append(upper_left_pix_y)
            print("Cluster Num: {} | X: {} | Y: {}".format(cluster_num, upper_left_pix_x, upper_left_pix_y))

        print("Done finding upper left pixel of each cluster...")
        print()

        # ========================================================================

        print("Creating minion images...")

        scr_shot.capture_sub_image(im, im_name, upper_left_pix_x_list, upper_left_pix_y_list, directory, 75)

        print("Done creating minion images...")
        print()

        # ========================================================================


        print("Creating New Image...")

        scr_shot.output_clusters_to_image(im, im_name, fil_x_clusters, fil_y_clusters, directory)

        print("Done Creating New Image...")
        print()

        #test