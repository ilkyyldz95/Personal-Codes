from __future__ import absolute_import
from __future__ import print_function
import pickle
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from scipy.misc import imresize
from os import listdir

class importData(object):
    def __init__(self, input_shape=(3, 224, 224), dir="./IMAGE_QUALITY_DATA"):
        self.input_shape = input_shape
        self.dir = dir

    def load_train_data(self):
        '''
        all_comp_labels: (ref_image, image1_name, image2_name, +1) is 1 > 2
        '''
        # load training data matrices
        all_comp_labels = np.load(self.dir + '/fac_train.npy')[:5000,:]
        #####################
        # reference image
        # load first image
        image_mtx = img_to_array(load_img(self.dir + all_comp_labels[0, 0])).astype(np.uint8)
        image_mtx = np.reshape(imresize(image_mtx, self.input_shape[1:]), self.input_shape)
        ref_imgs = image_mtx[np.newaxis, :, :, :]
        # load images
        for row in np.arange(1, all_comp_labels.shape[0]):
            image_mtx = img_to_array(load_img(self.dir + all_comp_labels[row, 0])).astype(np.uint8)
            image_mtx = np.reshape(imresize(image_mtx, self.input_shape[1:]), self.input_shape)[np.newaxis, :, :, :]
            ref_imgs = np.concatenate((ref_imgs, image_mtx), axis=0)
        # normalize
        ref_imgs = ref_imgs / 255
        # augmentation
        # ref_imgs_flip = np.flip(ref_imgs, axis=2)
        # ref_imgs = np.concatenate((ref_imgs, ref_imgs_flip), axis=0)
        # comparison images left
        # load first image
        image_mtx = img_to_array(load_img(self.dir + all_comp_labels[0, 1])).astype(np.uint8)
        image_mtx = np.reshape(imresize(image_mtx, self.input_shape[1:]), self.input_shape)
        comp_imgs_1 = image_mtx[np.newaxis, :, :, :]
        # load images
        for row in np.arange(1, all_comp_labels.shape[0]):
            image_mtx = img_to_array(load_img(self.dir + all_comp_labels[row, 1])).astype(np.uint8)
            image_mtx = np.reshape(imresize(image_mtx, self.input_shape[1:]), self.input_shape)[np.newaxis, :, :, :]
            comp_imgs_1 = np.concatenate((comp_imgs_1, image_mtx), axis=0)
        # normalize
        comp_imgs_1 = comp_imgs_1 / 255
        # augmentation
        # comp_imgs_1_flip = np.flip(comp_imgs_1, axis=2)
        # comp_imgs_1 = np.concatenate((comp_imgs_1, comp_imgs_1_flip), axis=0)
        # comparison images right
        # load first image
        image_mtx = img_to_array(load_img(self.dir + all_comp_labels[0, 2])).astype(np.uint8)
        image_mtx = np.reshape(imresize(image_mtx, self.input_shape[1:]), self.input_shape)
        comp_imgs_2 = image_mtx[np.newaxis, :, :, :]
        # load images
        for row in np.arange(1, all_comp_labels.shape[0]):
            image_mtx = img_to_array(load_img(self.dir + all_comp_labels[row, 2])).astype(np.uint8)
            image_mtx = np.reshape(imresize(image_mtx, self.input_shape[1:]), self.input_shape)[np.newaxis, :, :, :]
            comp_imgs_2 = np.concatenate((comp_imgs_2, image_mtx), axis=0)
        # normalize
        comp_imgs_2 = comp_imgs_2 / 255
        # augmentation
        # comp_imgs_2_flip = np.flip(comp_imgs_2, axis=2)
        # comp_imgs_2 = np.concatenate((comp_imgs_2, comp_imgs_2_flip), axis=0)
        # get corresponding labels
        # comp_labels = np.tile(all_comp_labels[:, 3].astype(int), (2,))
        comp_labels = all_comp_labels[:, 3].astype(int)
        return ref_imgs, comp_imgs_1, comp_imgs_2, comp_labels

    def load_test_data(self):
        '''
        all_abs_labels: (filtered image name, row['class']). sorted by reference image
        '''
        # For each reference image, get 22 filters sorted
        # load training data matrices
        all_abs_labels = np.load(self.dir + '/fac_test.npy')
        #####################
        image_mtx = img_to_array(load_img(self.dir + all_abs_labels[0, 0])).astype(np.uint8)
        image_mtx = np.reshape(imresize(image_mtx, self.input_shape[1:]), self.input_shape)
        abs_imgs = image_mtx[np.newaxis, :, :, :]
        # load images
        for row in np.arange(1, all_abs_labels.shape[0]):
            image_mtx = img_to_array(load_img(self.dir + all_abs_labels[row, 0])).astype(np.uint8)
            image_mtx = np.reshape(imresize(image_mtx, self.input_shape[1:]), self.input_shape)[np.newaxis, :, :, :]
            abs_imgs = np.concatenate((abs_imgs, image_mtx), axis=0)
        # get corresponding labels
        abs_labels = all_abs_labels[:, 1].astype(int)
        return abs_imgs, abs_labels

    def create_partitions(self):
        train_data_size = 1120
        # Separate reference images into training and testing
        np.random.seed(1)
        ref_image_names = [imgId for imgId in listdir(self.dir + '/Origin')]
        train_ref_image_names = ref_image_names[:train_data_size]
        test_ref_image_names = ref_image_names[train_data_size:]
        # Read files
        abs_label_file = "/image_score.pkl"
        comp_label_file = "/pairwise_comparison.pkl"
        with open(self.dir + comp_label_file, 'rb') as f:
            comp_label_matrix = pickle.load(f)
        with open(self.dir + abs_label_file, 'rb') as f:
            abs_label_matrix = pickle.load(f)
        #####################
        # get all comparison labels in training
        train_comp_labels = []
        for row in comp_label_matrix:
            # category, f1, f2, workerID, passDup, imgId, ans
            image1_name = '/' + row['f1'] + '/' + row['imgId'] + '.jpg'
            image2_name = '/' + row['f2'] + '/' + row['imgId'] + '.jpg'
            ref_image = '/Origin/' + row['imgId'] + '.jpg'
            if row['imgId'] + '.jpg' in train_ref_image_names:
                # save comparison label
                if row['ans'] == 'left':
                    train_comp_labels.append((ref_image, image1_name, image2_name, +1))
                elif row['ans'] == 'right':
                    train_comp_labels.append((ref_image, image1_name, image2_name, -1))
        # get all absolute labels in testing
        temp = []
        test_abs_labels = []
        # get all absolute labels in testing
        for row in abs_label_matrix:
            image_name = row['imgId'] + '.jpg'
            if image_name in test_ref_image_names:
                # filterName, imgId, class, score
                temp.append(('/' + row['filterName'] + '/', image_name, row['class']))
        # sort by reference image name
        temp = np.sort(temp, axis=1)
        for row in range(temp.shape[0]):
            # filtered image name, absolute label
            test_abs_labels.append((temp[row, 0] + temp[row, 2], temp[row, 1]))
        test_abs_labels = np.array(test_abs_labels)
        train_comp_labels = np.array(train_comp_labels)
        np.save('fac_train', train_comp_labels)
        np.save('fac_test', test_abs_labels)
