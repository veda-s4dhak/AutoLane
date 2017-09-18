import os
import pyscreenshot
from PIL import Image as PIL_image

#=======================================================================================================================

def create_dir(image_number):
    directory = '/Users/arnavgupta/PycharmProjects/LeagueOfLegendsAI/{}'.format(image_number)

    if not os.path.exists(directory):
        os.makedirs(directory)
        
#=======================================================================================================================

def get_screenshots(num_of_screenshots,directory,file_name):
    for i in range(1, num_of_screenshots+1):
        im = pyscreenshot.grab()
        im.save(directory+file_name+'{}.png'.format(i))

#=======================================================================================================================

def print_color_pixels(im):
    rgb_im = im.convert()
    x_max, y_max = rgb_im.size

    for x in range(1, x_max):
        for y in range(1, y_max):
            print( rgb_im.getpixel((x,y)) )

#=======================================================================================================================

def find_red_pixels(im):
    rgb_im = im.convert()
    x_max, y_max = rgb_im.size

    red_pixel_x_coordinate = []
    red_pixel_y_coordinate = []
    coordinate_cnt = 0

    for x in range (1,x_max):
        for y in range(1,y_max):
            r, g, b, a = rgb_im.getpixel((x,y))

            if (r > 150) and (g < 100) and (b < 100):
                red_pixel_x_coordinate.append(x)
                red_pixel_y_coordinate.append(y)
                coordinate_cnt+= 1

    return red_pixel_x_coordinate , red_pixel_y_coordinate , coordinate_cnt

#=======================================================================================================================

def find_clusters(x_coordinates,y_coordinates):
    cluster_x_list = []
    cluster_y_list = []

    while (len(x_coordinates) > 0):

        #=========Initializing a new cluster=========#
        cluster_x = []
        cluster_y = []
        cluster_x.append(x_coordinates[0])
        cluster_y.append(y_coordinates[0])
        del x_coordinates[0]
        del y_coordinates[0]

        # =========Creating a cluster=========#
        i = 0
        while (i < len(x_coordinates)):
            for j in range(0,len(cluster_x)):

                if (i >= len(x_coordinates)): #Need to break because while loop condition is not checked in for loop
                    break

                if ( is_pixel_adjacent( x_coordinates[i],y_coordinates[i],cluster_x[j],cluster_y[j] ) == True ):
                    # print("Condition checked: ( abs({}-{}) <= 10 ) and ( abs({}-{}) <= 10 ) == {}".format(
                    #     x_coordinates[i], cluster_x[j],y_coordinates[i], cluster_y[j],
                    #       is_pixel_adjacent(x_coordinates[i], y_coordinates[i], cluster_x[j], cluster_y[j]) ))

                    cluster_x.append(x_coordinates[i])
                    cluster_y.append(y_coordinates[i])
                    del x_coordinates[i]
                    del y_coordinates[i]
                    i = 0
                else:
                    # print("Condition checked: ( abs({}-{}) <= 10 ) and ( abs({}-{}) <= 10 ) == {}".format(
                    #     x_coordinates[i], cluster_x[j], y_coordinates[i], cluster_y[j],
                    #     is_pixel_adjacent(x_coordinates[i], y_coordinates[i], cluster_x[j], cluster_y[j])))

                    i += 1

                # print("i: {}".format(i))
                # print("Current X Cluster: {}".format(cluster_x))
                # print("Current Y Cluster: {}".format(cluster_y))
                # print("Current X Coord: {}".format(x_coordinates))
                # print("Current Y Coord: {}".format(y_coordinates))
                # input("Press any key to continue...")

        cluster_x_list.append(cluster_x)
        cluster_y_list.append(cluster_y)

        # print("Current X Cluster: {}".format(cluster_x))
        # print("Current Y Cluster: {}".format(cluster_y))
        # print("Size X Coord: {}".format(len(x_coordinates)))
        # print("-------------------------------------------------------------------------------------------------------")
        # input("Press Enter to continue...")

    return cluster_x_list , cluster_y_list

#=======================================================================================================================

def is_pixel_adjacent(x1,y1,x2,y2):

    adjacent = False

    if ( ( abs(x1 - x2) <= 40) and ( abs(y1 - y2) <= 20) ):
        adjacent = True

    return adjacent

#=======================================================================================================================

def output_clusters_to_image(im,x_clusters,y_clusters):
    pixel_map = im.load()

    color_cnt = 0;
    color = 255;

    for cluster_num in range( 0,len(x_clusters) ):

        for coord in range( 0,len(x_clusters[cluster_num]) ):
            if (color_cnt == 0):
                pixel_map[ x_clusters[cluster_num][coord],y_clusters[cluster_num][coord] ] = (color, 255, 255, 255)
            if (color_cnt == 1):
                pixel_map[ x_clusters[cluster_num][coord],y_clusters[cluster_num][coord] ] = (255, color, 255, 255)
            if (color_cnt == 2):
                pixel_map[ x_clusters[cluster_num][coord],y_clusters[cluster_num][coord] ] = (255, 255, color, 255)

        # Changing colors for each cluster
        color_cnt+=1

        if (color_cnt >= 3):
            color_cnt = 0;

        color = color-50
        if (color <= 0):
            color = 255;


    # Creating the new image
    x_max,y_max = im.size

    new_img = PIL_image.new(im.mode, im.size)
    new_pixel_map = new_img.load()

    for x in range(1,x_max):
        for y in range(1,y_max):
            new_pixel_map[x,y] = pixel_map[x,y]

    im.close()
    new_img.save(r'C:\Users\Veda Sadhak\Google Drive\A Different Time\AI Projects\Minion_Detection\Health_Bar\hb_detect.png')
    new_img.close()

#=======================================================================================================================

def filter_cluster_by_size(x_clusters,y_clusters,size = 50):


    for cluster_num in range(0,len(x_clusters)):

        if len(x_clusters[cluster_num]) < size:
            del x_clusters[cluster_num]

        if len(y_clusters[cluster_num]) < size:
            del y_clusters[cluster_num]

        if (cluster_num >= len(x_clusters)):
            break

    return x_clusters , y_clusters;
    
#=======================================================================================================================

def find_upper_left_pixel(x_cluster,y_cluster):
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
   
#=======================================================================================================================

def get_minion_area(im, cluster_x, cluster_y, size = 75):

    if type(cluster_x) == int:
        cluster_x = [cluster_x]
        cluster_y = [cluster_y]

    for i in xrange(0, len(cluster_x)):
        try:
            newImage = im.crop((cluster_x[i], cluster_y[i], cluster_x[i] + size, cluster_y[i] + size))
            newImage.save('/Users/arnavgupta/PycharmProjects/LeagueOfLegendsAI/{}/minion_{}.png'.format(im_number, i + 1),
                          'PNG')
        except:
            pass

#=======================================================================================================================

if __name__ == '__main__':
    #get_screenshots(5,r'C:\\Users\\Veda Sadhak\\Google Drive\\A Different Time\\AI Projects\\Minion_Detection\\Image_Set\\',r'league')
    
    im_number = 104
    create_dir(im_number)
    
    im = PIL_image.open(r'C:\Users\Veda Sadhak\Google Drive\A Different Time\AI Projects\Minion_Detection\Image_Set\league{}.png'.format(im_number))

    print("Finding Red Pixels...")

    r_pix_x_coord , r_pix_y_coord , coord_cnt = find_red_pixels(im)

    print("Red Pixel X Coordinates: {}".format(r_pix_x_coord))
    print("Red Pixel Y Coordinates: {}".format(r_pix_y_coord))
    print("Coordinate Count: {}".format(coord_cnt))

    print("Done Finding Red Pixels...")
    print()

   #========================================================================

    print("Finding Clusters...")

    x_clusters , y_clusters = find_clusters(r_pix_x_coord,r_pix_y_coord)

    print("X Clusters: {}".format(x_clusters))
    print("Size X Cluster: {}".format(len(x_clusters)))
    print("Y Clusters: {}".format(y_clusters))
    print("Size Y Cluster: {}".format(len(y_clusters)))

    print("Done Finding Clusters...")
    print()
    
    # ========================================================================

    print("Filtering Clusters...")

    fil_x_clusters, fil_y_clusters = filter_cluster_by_size(x_clusters, y_clusters,50)

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
        upper_left_pix_x, upper_left_pix_y = find_upper_left_pixel(fil_x_clusters[cluster_num],fil_y_clusters[cluster_num])
        upper_left_pix_x_list.append(upper_left_pix_x)
        upper_left_pix_y_list.append(upper_left_pix_y)
        print("Cluster Num: {} | X: {} | Y: {}".format(cluster_num, upper_left_pix_x, upper_left_pix_y))

    print("Done finding upper left pixel of each cluster...")
    print()

    # ========================================================================

    print("Creating minion images...")

    get_minion_area(im, upper_left_pix_x_list, upper_left_pix_y_list, size=75)

    print("Done creating minion images...")
    print()

    # ========================================================================

    print("Creating New Image...")

    output_clusters_to_image(im, fil_x_clusters, fil_y_clusters)

    print("Done Creating New Image...")
    print()
