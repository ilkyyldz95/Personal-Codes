# -*- coding: utf-8 -*-
import argparse
from os import listdir

import numpy as np
from scipy.ndimage import imread
from sklearn.model_selection import KFold  # import KFold

from inception_2nd_projection import inception_2nd_projection

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('findex', help='Index of test fold in set {0...4}')
    args = parser.parse_args()

    '''Parameters and locations'''
    # Input image dimensions for InceptionV3
    input_shape = (335, 472, 3)
    # Positive class data - guns
    data_path_pos = 'DATASET/positive/'
    # negative class data - random
    data_path_neg = 'DATASET/negative/'
    # Number of images in each class
    num_imag_pos = len(listdir(data_path_pos)) - len(listdir(data_path_pos))/2
    num_imag_neg = len(listdir(data_path_neg)) - len(listdir(data_path_neg))/2
    # List of all image names in the dataset
    all_images = listdir(data_path_pos)[len(listdir(data_path_pos))/2:] + \
                 listdir(data_path_neg)[len(listdir(data_path_neg))/2:]

    # Number of folds
    K = 5

    '''Initialize data'''
    print 'Starting k fold calculations...\n'
    # Initialize K-fold operator
    kf = KFold(n_splits=K, shuffle=True, random_state=1) # Define the split - into K folds
    # train_index, test_index = list(kf.split(range(num_imag_pos+num_imag_neg)))[int(args.findex)]
    train_index, test_index = list(kf.split(all_images))[int(args.findex)]


    x_train = np.zeros((len(train_index),335,472,3), dtype=np.uint8)
    y_train = np.zeros(len(train_index), dtype=np.uint8)

    x_test = np.zeros((len(test_index),335,472,3), dtype=np.uint8)
    y_test = np.zeros(len(test_index), dtype=np.uint8)

    Y = [1]*num_imag_pos + [0]*num_imag_neg

    print 'Starting to read images...\n'
    # Read images and place in X matrix
    count = 0
    for idx in train_index:

        if idx < num_imag_pos:
            temp = imread(data_path_pos + all_images[idx]).astype(np.uint8)
        else:
            temp = imread(data_path_neg + all_images[idx]).astype(np.uint8)

        if temp.shape != input_shape:
            temp = np.tile(temp[:, :, np.newaxis], (1, 1, 3))
        x_train[count, :, :, :] = temp
        y_train[count] = Y[idx]
        count += 1

    count = 0
    for idx in test_index:

        if idx < num_imag_pos:
            temp = imread(data_path_pos + all_images[idx]).astype(np.uint8)
        else:
            temp = imread(data_path_neg + all_images[idx]).astype(np.uint8)

        if temp.shape != input_shape:
            temp = np.tile(temp[:, :, np.newaxis], (1, 1, 3))
        x_test[count, :, :, :] = temp
        y_test[count] = Y[idx]
        count += 1

        print 'Passing 2nd half of data to inception...\n'
        accuracy, training_duration = inception_2nd_projection(x_train, y_train, x_test, y_test, save_model=args.findex)