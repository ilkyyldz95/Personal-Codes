from os import listdir
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imresize, imsave
from scipy.ndimage import rotate
#import matplotlib.pyplot as plt

image_dir = './negative/'

output_dir = './negative_rotated/'

# check if dir is correct
# print listdir(image_dir)
# initial test values
# highest_shape = [0,0]
# smallest_shape = [400,400]
# sum of dimensions of all images
# dim_sum = [0,0]
# number of images
# num_images =  len(listdir(image_dir))

dim_average = [335, 472]

for image_name in listdir(image_dir):

    # read image
    img = imread(image_dir+image_name)

    # rotate image 4 times
    img_rotate_90 = rotate(img, 90, axes=(0, 1))
    img_rotate_180 = rotate(img_rotate_90, 90, axes=(0, 1))
    img_rotate_270 = rotate(img_rotate_180, 90, axes=(0, 1))

    # calculate smallest and highest dimensions
    # dim_sum[0] += img.shape[0]
    # dim_sum[1] += img.shape[1]
    # if img.shape[0]*img.shape[1] > highest_shape[0]*highest_shape[1]:
    #     highest_shape[0] = img.shape[0]
    #     highest_shape[1] = img.shape[1]
    # if img.shape[0] * img.shape[1] < smallest_shape[0] * smallest_shape[1]:
    #     smallest_shape[0] = img.shape[0]
    #     smallest_shape[1] = img.shape[1]

    # reshape image
    # resized_image = imresize(img, dim_average, interp='bilinear', mode=None)

    # save image
    imsave(output_dir + '_1_' + image_name, img_rotate_90)
    imsave(output_dir + '_2_' + image_name, img_rotate_180)
    imsave(output_dir + '_3_' + image_name, img_rotate_270)

    # # Display images if you want to
    # plt.imshow(np.uint8(img))
    # plt.show()
    # plt.imshow(np.uint8(resized_image))
    # plt.show()

# dim_average = np.array(dim_sum)/num_images
# print 'Smallest:', smallest_shape
# print 'Highest:', highest_shape
# print 'Average:',dim_average
# print 'Last images shape:', resized_image.shape




