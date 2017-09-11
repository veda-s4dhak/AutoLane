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
        for j in range(1, y_max):
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

                #print("i: {} | j: {}".format(i,j))

                if ( is_pixel_adjacent( x_coordinates[i],y_coordinates[i],cluster_x[j],cluster_y[j] ) == True ):
                    cluster_x.append(x_coordinates[i])
                    cluster_y.append(y_coordinates[i])
                    del x_coordinates[i]
                    del y_coordinates[i]

            i += 1

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

    if ( ( abs(x1 - x2) <= 1) and ( abs(y1 - y2) <= 1) ):
        adjacent = True

    return adjacent

#=======================================================================================================================

def output_clusters_to_image(im,x_clusters,y_clusters):
    pixel_map = im.load()

    for cluster_num in range( 0,len(x_clusters) ):
        print("Current Cluster Length: {}".format( len(x_clusters[cluster_num]) ))

        if ( len(x_clusters[cluster_num]) > 25 ):
            for coord in range( 0,len(x_clusters[cluster_num]) ):
                pixel_map[ x_clusters[cluster_num][coord],y_clusters[cluster_num][coord] ] = (255, 255, 255, 255)

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
    
    im = PIL_image.open(r'C:\Users\Veda Sadhak\Google Drive\A Different Time\AI Projects\Minion_Detection\Image_Set\league{}.png'.format(im_number)

    print("Finding Red Pixels...")

    r_pix_x_coord , r_pix_y_coord , coord_cnt = find_red_pixels(im)

    print("Red Pixel X Coordinates: {}".format(r_pix_x_coord))
    print("Red Pixel Y Coordinates: {}".format(r_pix_y_coord))
    print("Coordinate Count: {}".format(coord_cnt))

    print("Done Finding Red Pixels...")
    print()

    print("Finding Clusters...")

    x_clusters , y_clusters = find_clusters(r_pix_x_coord,r_pix_y_coord)

    print("X Clusters: {}".format(x_clusters))
    print("Size X Cluster: {}".format(len(x_clusters)))
    print("Y Clusters: {}".format(y_clusters))
    print("Size Y Cluster: {}".format(len(y_clusters)))

    print("Done Finding Clusters...")
    print()

    print("Creating New Image...")

    output_clusters_to_image(im, x_clusters, y_clusters)

    print("Done Creating New Image...")
    print()
