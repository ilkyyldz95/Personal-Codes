from __future__ import absolute_import
from __future__ import print_function
import sys
import numpy as np
import pickle
from keras.preprocessing.image import load_img, img_to_array
from scipy.ndimage import rotate

# Import 100 images
partition_file_100 = pickle.load(open('./Partitions.p', 'rb'))
'''with open('./Partitions.p', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    partition_file_100 = u.load()'''
img_folder_100 = './preprocessed/All/'

# Import 6000 images
partition_file_6000 = pickle.load(open('./6000Partitions.p', 'rb'))
'''with open('./6000Partitions.p', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    partition_file_6000 = u.load()'''
img_folder_6000 = 'preprocessed_JamesCode/'

class importData(object):
    def __init__(self, kthFold):
        self.kthFold = kthFold

    def biLabels(self,labels):
        """
        This function will binarize labels.
        There are C classes {1,2,3,4,...,c} in the labels, the output would be c dimensional vector.
        Input:
            - labels: (N,) np array. The element value indicates the class index.
        Output:
            - biLabels: (N, C) array. Each row has and only has a 1, and the other elements are all zeros.
        Example:
            The input labels = np.array([1,2,2,1,3])
            The binaried labels are np.array([[1,0,0],[0,1,0],[0,1,0],[1,0,0],[0,0,1]])
        """
        N = labels.shape[0]
        labels.astype(np.int)
        C = len(np.unique(labels))
        binarized = np.zeros((N, C))
        binarized[np.arange(N).astype(np.int), labels.astype(np.int)] = 1
        return binarized

    def importAbsTrainData(self):
        # LOAD DATA FOR ABSOLUTE LABELS
        part_rsd_train = partition_file_100['RSDTrainPlusPartition']
        label_absolute = partition_file_100['label13']
        label_absolute[label_absolute==1.5] = 2
        order_name = partition_file_100['orderName']
        for k in [self.kthFold]:
            k_ind_rsd_train = part_rsd_train[k]
            k_img_train_list = [img_folder_100+order_name[order+1]+'.png' for order in k_ind_rsd_train]
            # Load Images
            # Image for training
            k_img_train = img_to_array(load_img(k_img_train_list[0]))[np.newaxis,:,:,:]
            for img_name_iter in k_img_train_list[1:]:
                img_iter = img_to_array(load_img(img_name_iter))[np.newaxis,:,:,:]
                k_img_train = np.concatenate((k_img_train,img_iter),axis=0)
            k_img_train_ori = np.tile(k_img_train,[13,1,1,1])
            k_label_abs_train = label_absolute[k_ind_rsd_train,:]
            k_label_train_ori = np.reshape(k_label_abs_train,(-1,),order='F')
            k_label_train_bi = self.biLabels(np.array(k_label_train_ori))
            # Rotate image at each 90 degree and flip images. A single image will show 8 times via its rotations and filps in training.
            k_img_train_rotate_90 = rotate(k_img_train_ori,90,axes=(2,3))
            k_img_train_rotate_90_ori = np.concatenate((k_img_train_ori,k_img_train_rotate_90),axis=0)
            k_img_train_rotate_180 =  rotate(k_img_train_rotate_90_ori,180,axes=(2,3))
            k_img_train_rotate = np.concatenate((k_img_train_rotate_90_ori,k_img_train_rotate_180),axis=0)
            k_img_train_flip = np.flipud(k_img_train_rotate)
            k_img_train = np.concatenate((k_img_train_rotate,k_img_train_flip),axis=0)
            k_label_train = 1.0 * np.tile(k_label_train_bi,(8,1))

        return k_img_train, k_label_train

    def importAbsTestData(self):
        part_rsd_test = partition_file_100['RSDTestPlusPartition']
        label_absolute = partition_file_100['label13']
        label_absolute[label_absolute == 1.5] = 2
        order_name = partition_file_100['orderName']
        for k in [self.kthFold]:
            k_ind_rsd_test = part_rsd_test[k]
            k_img_test_list = [img_folder_100 + order_name[order + 1] + '.png' for order in k_ind_rsd_test]
            # Load Images
            k_img_test = img_to_array(load_img(k_img_test_list[0]))[np.newaxis, :, :, :]
            for img_name_iter in k_img_test_list[1:]:
                img_iter = img_to_array(load_img(img_name_iter))[np.newaxis, :, :, :]
                k_img_test = np.concatenate((k_img_test, img_iter), axis=0)
            k_img_test = np.tile(k_img_test, [13, 1, 1, 1])
            k_label_abs_test = label_absolute[k_ind_rsd_test, :]
            k_label_test = 1.0 * np.reshape(k_label_abs_test, (-1,), order='F')

        return k_img_test, k_label_test

    def importCompTrainData(self, data_size=None):
        part_cmp_train = partition_file_100['cmpTrainPlusPartition']  # (fold,pair index)
        label_cmp = partition_file_100['cmpData']  # (expert,imagename1,imagename2,label)
        k_ind_image_all = label_cmp[0]  # (imagename1,imagename2, label)
        for k in [self.kthFold]:
            # get all images and labels in kth fold
            k_ind_cmp_train = part_cmp_train[k]  # indices corresponding to label_all
            # tuples of compared image names
            k_img_train_list = [
                (img_folder_100 + k_ind_image_all[index][0] + '.png', img_folder_100 + k_ind_image_all[index][1] + '.png') for
                index in k_ind_cmp_train]
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

            # Rotate image at each 90 degree and flip images. A single image will show 8 times via its rotations and filps in training.
            k_img_train_1_rotate_90 = rotate(k_img_train_ori_1,90,axes=(2,3)).astype(np.uint8)
            k_img_train_1_rotate_90_ori = np.concatenate((k_img_train_ori_1,k_img_train_1_rotate_90),axis=0)
            k_img_train_1_rotate_180 = rotate(k_img_train_1_rotate_90_ori,180,axes=(2,3)).astype(np.uint8)
            k_img_train_1 = np.concatenate((k_img_train_1_rotate_90_ori,k_img_train_1_rotate_180),axis=0)
            #k_img_train_1_flip = np.flipud(k_img_train_1_rotate).astype(np.uint8)
            #k_img_train_1 = np.concatenate((k_img_train_1_rotate,k_img_train_1_flip),axis=0)

            k_img_train_2_rotate_90 = rotate(k_img_train_ori_2, 90, axes=(2, 3)).astype(np.uint8)
            k_img_train_2_rotate_90_ori = np.concatenate((k_img_train_ori_2, k_img_train_2_rotate_90), axis=0)
            k_img_train_2_rotate_180 = rotate(k_img_train_2_rotate_90_ori, 180, axes=(2, 3)).astype(np.uint8)
            k_img_train_2 = np.concatenate((k_img_train_2_rotate_90_ori, k_img_train_2_rotate_180), axis=0)
            #k_img_train_2_flip = np.flipud(k_img_train_2_rotate).astype(np.uint8)
            #k_img_train_2 = np.concatenate((k_img_train_2_rotate, k_img_train_2_flip), axis=0)

            k_label_train = 1.0 * np.transpose(np.tile(k_label_comp_train, (1, 4)))
            #k_label_train = np.transpose(np.tile(k_label_comp_train,(1,8)))

        if data_size != None:
            ###########################
            # Shuffle and choose images
            # ONLY WHEN NO ROTATIONS
            indices = np.arange(data_size)
            np.random.seed(1)
            np.random.shuffle(indices)
            k_img_train_1 = k_img_train_1[indices]
            k_img_train_2 = k_img_train_2[indices]
            k_label_train = k_label_train[indices]
            ###########################

        return k_img_train_1, k_img_train_2, k_label_train

    def importCompTestData(self):
        part_cmp_test = partition_file_100['cmpTestPlusPartition']  # (fold,pair index)
        label_cmp = partition_file_100['cmpData']  # (expert,imagename1,imagename2,label)
        k_ind_image_all = label_cmp[0]  # (imagename1,imagename2, label)
        for k in [self.kthFold]:
            # get all images and labels in kth fold
            k_ind_cmp_test = part_cmp_test[k]  # indices corresponding to label_all
            # tuples of compared image names
            k_img_test_list = [
                (img_folder_100 + k_ind_image_all[index][0] + '.png', img_folder_100 + k_ind_image_all[index][1] + '.png') for
                index in k_ind_cmp_test]
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
            k_label_test = 1.0 * np.array([label_cmp[l][index][2] for index in k_ind_cmp_test for l in range(5)])

        return k_img_test_1, k_img_test_2, k_label_test

    def importAbsTrain6000Data(self, data_size=None):
        # LOAD DATA FOR ABSOLUTE LABELS
        part_rsd_train = partition_file_6000['RSDTrainPartition']
        label_rsd = partition_file_6000['RSDLabels']
        img_names = partition_file_6000['imgNames']
        for k in [self.kthFold]:
            k_ind_rsd_train = part_rsd_train[k].astype(np.int)
            k_img_train_list = [img_folder_6000 + img_names[int(order)] + '.png' for order in k_ind_rsd_train]
            # Load Labels
            k_label_abs_train = label_rsd[k_ind_rsd_train] - 1  # make 012
            k_label_train_bi = 1.0 * self.biLabels(np.array(k_label_abs_train))
            # Load Images
            k_img_train_ori = img_to_array(load_img(k_img_train_list[0]))[np.newaxis, :, :, :]
            for img_name_iter in k_img_train_list[1:]:
                img_iter = img_to_array(load_img(img_name_iter))[np.newaxis, :, :, :]
                k_img_train_ori = np.concatenate((k_img_train_ori, img_iter), axis=0)
            # Rotate image at each 90 degree and flip images. A single image will show 8 times via its rotations and filps in training.
            k_img_train_rotate_90 = rotate(k_img_train_ori, 90, axes=(2, 3))
            k_img_train_rotate_90_ori = np.concatenate((k_img_train_ori, k_img_train_rotate_90), axis=0)
            k_img_train_rotate_180 = rotate(k_img_train_rotate_90_ori, 180, axes=(2, 3))
            k_img_train_rotate = np.concatenate((k_img_train_rotate_90_ori, k_img_train_rotate_180), axis=0)
            k_img_train_flip = np.flipud(k_img_train_rotate)
            k_img_train = np.concatenate((k_img_train_rotate, k_img_train_flip), axis=0)
            k_label_train = 1.0 * np.tile(k_label_train_bi, (8, 1))

        if data_size != None:
            ###########################
            # Shuffle and choose images
            # ONLY WHEN NO ROTATIONS
            indices = np.arange(data_size)
            np.random.seed(1)
            np.random.shuffle(indices)
            k_img_train = k_img_train[indices]
            k_label_train = k_label_train[indices]
            ###########################

        return k_img_train, k_label_train

    def importAbsTest6000Data(self):
        part_rsd_test = partition_file_6000['RSDTestPartition']
        label_rsd = partition_file_6000['RSDLabels']
        img_names = partition_file_6000['imgNames']
        for k in [self.kthFold]:
            k_ind_rsd_test = part_rsd_test[k].astype(np.int)
            k_img_test_list = [img_folder_6000 + img_names[int(order)] + '.png' for order in k_ind_rsd_test]
            # Load Images
            k_img_test_ori = img_to_array(load_img(k_img_test_list[0]))[np.newaxis, :, :, :]
            for img_name_iter in k_img_test_list[1:]:
                img_iter = img_to_array(load_img(img_name_iter))[np.newaxis, :, :, :]
                k_img_test_ori = np.concatenate((k_img_test_ori, img_iter), axis=0)
            k_label_abs_test = 1.0 * label_rsd[k_ind_rsd_test] - 1 #make 012

        return k_img_test_ori, k_label_abs_test