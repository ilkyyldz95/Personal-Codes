import pickle
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Lambda, Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras import backend as K
import numpy as np
from basenet_functional import *
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from math import sqrt

# ONLY TRUE SIAMESE CODE SO FAR, BASE NET IS CREATED ONLY FOR ONCE

def BTPred(scalars):
    """
    This is the output when we predict comparison labels. s1, s2 = scalars (beta.*x)
    """
    s1 = scalars[0]
    s2 = scalars[1]
    return s1 - s2

def BTLoss(y_true, y_pred):
    """
    Negative log likelihood of Bradley-Terry Penalty, to be minimized. y = beta.*x
    y_true:-1 or 1
    y_pred:si-sj
    """
    exponent = K.exp(-y_true * (y_pred))
    return K.log(1 + exponent)

def diffLoss(y_true, y_pred):
    return -y_true * y_pred

def ThurstoneLoss(y_true, y_pred):
    '''P(si-sj|yij) = 0.5 * erfc(-yij*(si-sj) / sqrt(2))'''
    return -K.log(0.5 * tf.erfc(-y_true*y_pred / sqrt(2)))

class onlyComparisonNN(object):
    '''
    Properties and methods of the comparison CNN
    Imports googlenet architecture and uses it as the base network of Siamese architecture
    Trains only with comparisons, which only exist in first 100
    '''
    def __init__(self,partition_file_100='./Partitions.p', img_folder_100='./preprocessed/All/',
                 partition_file_6000='./6000Partitions.p', img_folder_6000='./preprocessed_JamesCode/'):
        """
        Find the cross validation partition_file_path and image folder path.
        partition_file_path: contains image names, cross-validation splits, and labels.
        img_folder_path: contains images with associated names in partition_file.
        """
        self.img_folder_100 = img_folder_100
        self.partition_file_100 = pickle.load(open(partition_file_100, 'rb'))
        self.img_folder_6000 = img_folder_6000
        self.partition_file_6000 = pickle.load(open(partition_file_6000, 'rb'))

    def choose_cmp_from_plus_or_not(self, partition_file_100, ind_cmp_train):
        """
        Find the comparison indices in which one image coming from plus category and another image coming from not plus category.
        Input:
            - partition_file_100: dictionary loading from pickle file of 100 images.
            - ind_cmp_train: the comparison indices in the pickle file, which will be filtered to contain only plus vs not plus comparisons.

        Output:
            - ind_cmp_train_plus_or_not: the comparison indices in where plus vs not plus comparison contained.
        """
        #        part_cmp_train = partition_file_100['cmpTrainPlusPartition']
        #        ind_cmp_train = part_cmp_train[0]
        label_plus_100 = partition_file_100['RSDLabels'].astype(np.int64)
        label_plus_100[np.where(label_plus_100 != 1)[0]] = -1
        img_name_100 = partition_file_100['orderName'].values()
        img_name_plus_label_pair = dict(zip(img_name_100, label_plus_100))
        cmp_data = partition_file_100['cmpData'][0]
        ind_cmp_train_plus_or_not = []
        for cmp_ind in ind_cmp_train:
            img_a, img_b, _ = cmp_data[cmp_ind]
            if img_name_plus_label_pair[img_a] != img_name_plus_label_pair[img_b]:
                ind_cmp_train_plus_or_not.append(cmp_ind)
        return ind_cmp_train_plus_or_not

    def load_training_cmp_data(self, kthFold, num_unique_images=12):
        """
            Load training images and labels for k-th fold with augmentation.
            drop_rate: float, [0.0,1.0], the percentage that kept in training. 1.0 means keep all data. NOT AN OPTION FOR NOW
            rotation: Boolean parameter indicates whether adding rotated images to the training set.
            flip: Boolean parameter indicates whether adding flipped images to the training set.
        """
        part_cmp_train = self.partition_file_100['cmpTrainPlusPartition']  # (fold,pair index)
        label_cmp = self.partition_file_100['cmpData']  # (expert,imagename1,imagename2,label)
        k_ind_image_all = label_cmp[0]  # (imagename1,imagename2, label)
        # get all images and labels in kth fold
        k_ind_cmp_train = part_cmp_train[kthFold]  # indices corresponding to label_all
        # choose number of unique images
        np.random.seed(1)
        num_unique_comb = int(np.floor(num_unique_images*(num_unique_images-1)/2))
        k_ind_cmp_train = k_ind_cmp_train[np.random.choice(len(k_ind_cmp_train), num_unique_comb, replace=False)]
        ################################only interclass comp data
        k_ind_cmp_train = self.choose_cmp_from_plus_or_not(self.partition_file_100, k_ind_cmp_train)
        ################################
        # tuples of compared image names
        k_img_train_list = [(self.img_folder_100 + k_ind_image_all[index][0] + '.png',
                            self.img_folder_100 + k_ind_image_all[index][1] + '.png') for index in k_ind_cmp_train]
        # Load Images
        # k_img_train_1: 1st elm of all pairs,channels,imagex,imagey
        k_img_train_1 = img_to_array(load_img(k_img_train_list[0][0])).astype(np.uint8)[np.newaxis, :, :, :]
        k_img_train_2 = img_to_array(load_img(k_img_train_list[0][1])).astype(np.uint8)[np.newaxis, :, :, :]
        for img_names_iter in k_img_train_list[1:]:
            img_iter_1 = img_to_array(load_img(img_names_iter[0])).astype(np.uint8)[np.newaxis, :, :, :]
            k_img_train_1 = np.concatenate((k_img_train_1, img_iter_1), axis=0)
            img_iter_2 = img_to_array(load_img(img_names_iter[1])).astype(np.uint8)[np.newaxis, :, :, :]
            k_img_train_2 = np.concatenate((k_img_train_2, img_iter_2), axis=0)
        # Replicate for all experts
        k_img_train_ori_1 = np.tile(k_img_train_1, [5, 1, 1, 1])
        k_img_train_ori_2 = np.tile(k_img_train_2, [5, 1, 1, 1])
        # Load labels
        k_label_comp_train = np.array([label_cmp[l][index][2] for index in k_ind_cmp_train for l in range(5)])
        return k_img_train_ori_1, k_img_train_ori_2, k_label_comp_train

    def load_comp_testing_data(self,kthFold, test_100=True):
        """
        Load testing images for k-th fold.
        kthFold: int, in [0,4]
        """
        if test_100:  # validation on kthFold
            part_cmp_test = self.partition_file_100['cmpTestPlusPartition']  # (fold,pair index)
            label_cmp = self.partition_file_100['cmpData']  # (expert,imagename1,imagename2,label)
            k_ind_image_all = label_cmp[0]  # (imagename1,imagename2, label)
            # get all images and labels in kth fold
            k_ind_cmp_test = part_cmp_test[kthFold]  # indices corresponding to label_all
            # tuples of compared image names
            k_img_test_list = [(self.img_folder_100 + k_ind_image_all[index][0] + '.png',
                                self.img_folder_100 + k_ind_image_all[index][1] + '.png')
                               for index in k_ind_cmp_test]
            # Load Images
            k_img_test_1 = img_to_array(load_img(k_img_test_list[0][0])).astype(np.uint8)[np.newaxis, :, :, :]
            k_img_test_2 = img_to_array(load_img(k_img_test_list[0][1])).astype(np.uint8)[np.newaxis, :, :, :]
            for img_names_iter in k_img_test_list[1:]:
                img_iter_1 = img_to_array(load_img(img_names_iter[0])).astype(np.uint8)[np.newaxis, :, :, :]
                k_img_test_1 = np.concatenate((k_img_test_1, img_iter_1), axis=0)
                img_iter_2 = img_to_array(load_img(img_names_iter[1])).astype(np.uint8)[np.newaxis, :, :, :]
                k_img_test_2 = np.concatenate((k_img_test_2, img_iter_2), axis=0)
            # Replicate for all experts
            k_img_test_1 = np.tile(k_img_test_1, [5, 1, 1, 1])
            k_img_test_2 = np.tile(k_img_test_2, [5, 1, 1, 1])
            k_label_test = np.array([label_cmp[l][index][2] for index in k_ind_cmp_test for l in range(5)])
        else:
            part_cmp_test = self.partition_file_100['cmpTestPlusPartition']  # (fold,pair index)
            label_cmp = self.partition_file_100['cmpData']  # (expert,imagename1,imagename2,label)
            k_ind_image_all = label_cmp[0]  # (imagename1,imagename2, label)
            # get all images and labels
            k_ind_cmp_test = part_cmp_test[0]  # indices corresponding to label_all
            for k in [1, 2, 3, 4]:
                k_ind_cmp_test = np.append(k_ind_cmp_test, part_cmp_test[k])
            # tuples of compared image names
            k_img_test_list = [(self.img_folder_100 + k_ind_image_all[index][0] + '.png',
                                self.img_folder_100 + k_ind_image_all[index][1] + '.png')
                               for index in k_ind_cmp_test]
            # Load Images
            k_img_test_1 = img_to_array(load_img(k_img_test_list[0][0])).astype(np.uint8)[np.newaxis, :, :, :]
            k_img_test_2 = img_to_array(load_img(k_img_test_list[0][1])).astype(np.uint8)[np.newaxis, :, :, :]
            for img_names_iter in k_img_test_list[1:]:
                img_iter_1 = img_to_array(load_img(img_names_iter[0])).astype(np.uint8)[np.newaxis, :, :, :]
                k_img_test_1 = np.concatenate((k_img_test_1, img_iter_1), axis=0)
                img_iter_2 = img_to_array(load_img(img_names_iter[1])).astype(np.uint8)[np.newaxis, :, :, :]
                k_img_test_2 = np.concatenate((k_img_test_2, img_iter_2), axis=0)
            # Replicate for all experts
            k_img_test_1 = np.tile(k_img_test_1, [5, 1, 1, 1])
            k_img_test_2 = np.tile(k_img_test_2, [5, 1, 1, 1])
            k_label_test = np.array([label_cmp[l][index][2] for index in k_ind_cmp_test for l in range(5)])
        return k_img_test_1, k_img_test_2, k_label_test

    def train(self, kthFold, init_weight='./googlenet_weights.h5', save_model_name='./onlyComp.h5',
              epochs=100, learning_rate=1e-4, num_unique_images=12, batch_size=24,
              comp_loss=BTLoss, num_of_classes=3, num_of_features=1024, score_layer=4):
        """
        Training CNN except k-th fold.
        kthFold: The fold that is going to be tested.
        save_model_name: The file contains the trained model.
        epochs: iterations of training.
        learning_rate:  learning rate for training.
        batch_size: batch size to train.
        loss: Loss function of the neural network
        score_layer: how many layers to learn single score from 1024 features
        """
        training_imgs_1, training_imgs_2, comp_training_labels = \
            self.load_training_cmp_data(kthFold, num_unique_images=num_unique_images)
        print("images loaded")
        # create the architecture
        input_a = Input(shape=(3, 224, 224))
        input_b = Input(shape=(3, 224, 224))
        comp_net = create_base_network(input_a, input_b, no_classes=num_of_classes, no_features=num_of_features,
                                           num_score_layer=score_layer)
        comp_net.load_weights(init_weight, by_name=True)
        comp_net.compile(optimizer=SGD(lr=learning_rate), loss=comp_loss)
        print("network created")
        # Train: Iteratively train each model at each epoch, with weight of alpha
        comp_net.fit([training_imgs_1, training_imgs_2], comp_training_labels, batch_size=batch_size, epochs=epochs)
        # Save weights
        comp_net.save(save_model_name)

    def test(self, kthFold, model_file, learning_rate=1e-4, comp_loss=BTLoss, num_of_classes=3,
             num_of_features=1024, score_layer=1, test_100=True, num_unique_images=12):
        """
        Testing CNN for kthFold images.
        kthFold: int, in [0,4]
        model_file: string, path of the trained model.
        """
        comp_img_test_1, comp_img_test_2, comp_label_test = self.load_comp_testing_data(kthFold, test_100)
        # create the architecture
        input_a = Input(shape=(3, 224, 224))
        input_b = Input(shape=(3, 224, 224))
        comp_net = create_base_network(input_a, input_b, no_classes=num_of_classes, no_features=num_of_features,
                                       num_score_layer=score_layer)
        # load trained model
        comp_net.load_weights(model_file)
        comp_net.compile(optimizer=SGD(lr=learning_rate), loss=comp_loss)
        # test comparisons
        y_comp_pred = comp_net.predict([comp_img_test_1, comp_img_test_2])

        with open('test_only_comp.txt', 'a') as file:
            file.write('\n\nFold: ' + str(kthFold))
            file.write('\nScore layers: ' + str(score_layer))
            file.write('\nLoss: ' + str(comp_loss))
            file.write('\nComparison AUC: '+str(roc_auc_score(comp_label_test, y_comp_pred)))







