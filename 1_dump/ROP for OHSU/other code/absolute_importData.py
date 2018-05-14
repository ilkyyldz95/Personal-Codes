from __future__ import absolute_import
from __future__ import print_function
import pickle
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

class importData(object):
    def __init__(self,partition_file_100='./Partitions.p', img_folder_100='./preprocessed/All/',
                 partition_file_6000='./6000Partitions.p', img_folder_6000='./preprocessed_JamesCode/'):
        """
        Find the cross validation partition_file_path and image folder path.
        partition_file_path: 6000.p contains image names, cross-validation splits, and labels.
        img_folder_path: it contains  5000 images with associated names in partition_file.
        """
        self.img_folder_100 = img_folder_100
        self.partition_file_100 = pickle.load(open(partition_file_100, 'rb'))
        self.img_folder_6000 = img_folder_6000
        self.partition_file_6000 = pickle.load(open(partition_file_6000, 'rb'))

        '''with open(partition_file_100, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            self.partition_file_100 = u.load()
        '''

    def obtain_comparison_indices(self, img_index_train, imgNames100, part_cmp_index, cmp_pair_names_experts):
        """
        This function is to get the training comparison data from the given indices of 1st 100 images.
        Input:
            - img_index_train: The index of 1st 100 images.
        Output:
            - cmp_indices:
            - cmp_labels:
        """
        train_rsd_img_names = [imgNames100[img_index + 1] for img_index in list(img_index_train)]
        cmp_indices = []
        for i in part_cmp_index:
            img_name_i, img_name_j = cmp_pair_names_experts[0][i][0], cmp_pair_names_experts[0][i][1]
            if img_name_i in train_rsd_img_names and img_name_j in train_rsd_img_names:
                cmp_indices.append(i)
        return cmp_indices

    def load_cmp_data(self, label_cmp, cmp_indices):
        """
            Load comparisons with given indices
            label_cmp: (expert,imagename1,imagename2,label)
        """
        # tuples of compared image names
        k_img_train_list = [(self.img_folder_100 + label_cmp[0][index][0] + '.png',
                            self.img_folder_100 + label_cmp[0][index][1] + '.png') for index in cmp_indices]
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
        k_label_comp_train = np.array([label_cmp[l][index][2] for l in range(5) for index in cmp_indices])
        return k_img_train_ori_1, k_img_train_ori_2, k_label_comp_train

    def biLabels(self, labels):
        """
        This function will binarized labels.
        There are C classes {1,2,3,4,...,c} in the labels, the output would be c dimensional vector.
        Input:
            - labels: (N,) np array. The element value indicates the class index.
        Output:
            - biLabels: (N, C) array. Each row has and only has a 1, and the other elements are all zeros.
            - C: integer. The number of classes in the data.
        Example:
            The input labels = np.array([1,2,2,1,3])
            The binaried labels are np.array([[1,0,0],[0,1,0],[0,1,0],[1,0,0],[0,0,1]])
        """
        N = labels.shape[0]
        labels.astype(np.int)
        C = len(np.unique(labels))
        binarized = np.zeros((N, C))
        binarized[np.arange(N).astype(np.int), labels.astype(np.int).reshape((N,)) - 1] = 1
        return binarized, C

    def load_training_data(self, balance=False, num_unique_images=80):
        """
        Load training images and labels for k-th fold with augmentation.
        loaded labels: 0:normal, 1: prep, 2:plus
        balance: if true, sample the same number of absolute labels from each class
        rotation: Boolean parameter indicates whether adding rotated images to the training set.
        flip: Boolean parameter indicates whether adding flipped images to the training set.
        Binary classification is assumed. abs_thr='plus' : only plus is 1. abs_thr='prep' : preplus and plus are 1.
        """
        # setup for absolute labels
        part_abs_test = self.partition_file_100['RSDTestPlusPartition']
        label_abs = self.partition_file_100['label13']
        img_names = self.partition_file_100['orderName']
        # setup for comparison labels
        part_cmp_train = self.partition_file_100['cmpTrainPlusPartition']  # (fold,pair index)
        label_cmp = self.partition_file_100['cmpData']  # (expert,imagename1,imagename2,label)
        ############################################################
        # load training images for folds 1,2,4
        ind_abs_train = part_abs_test[1].astype(np.int)
        ind_abs_train = np.append(ind_abs_train, part_abs_test[2].astype(np.int))
        ind_abs_train = np.append(ind_abs_train, part_abs_test[4].astype(np.int))
        #ind_abs_train = np.append(ind_abs_train, part_abs_test[0].astype(np.int))
        # load absolute labels
        img_train_list = [self.img_folder_100 + img_names[int(order + 1)] + '.png' for order in ind_abs_train]
        img_train_ori = img_to_array(load_img(img_train_list[0])).astype(np.uint8)[np.newaxis, :, :, :]
        for img_name_iter in img_train_list[1:]:
            img_iter = img_to_array(load_img(img_name_iter)).astype(np.uint8)[np.newaxis, :, :, :]
            img_train_ori = np.concatenate((img_train_ori, img_iter), axis=0)
        # Load training labels for kthFold
        label_fold = label_abs[ind_abs_train]
        # Balance the data from original images (not augmented images)
        ind_list = []
        len_ind_list = []
        for c_iter in range(3):
            ind_list.append(np.where(label_fold[:,0].astype(int) == c_iter)[0])
            len_ind_list.append(len(ind_list[c_iter]))
        np.random.seed(1)
        ind_balanced_list = []
        if not balance:
            # keep the ratio of classes while dropping
            for c_balance_iter in range(3):
                no_sample = int(np.floor(len_ind_list[c_balance_iter] * num_unique_images / sum(len_ind_list)))
                if no_sample < 1:
                    raise ValueError("Enter a higher number")
                else:
                    ind_balanced_list.append(
                        ind_list[c_balance_iter][
                            np.random.choice(len_ind_list[c_balance_iter], no_sample, replace=False)])
        else:
            min_unique_img_class = np.floor(1. * num_unique_images / 3)
            min_size_class = min(len_ind_list)
            if min_unique_img_class > min_size_class:
                raise ValueError("The maximum number of images for single class is: " + str(min_size_class))
            else:
                min_size_class = int(min_unique_img_class)
            for c_balance_iter in range(3):
                ind_balanced_list.append(
                    ind_list[c_balance_iter][
                        np.random.choice(len_ind_list[c_balance_iter], min_size_class, replace=False)])
        img_train_index = np.concatenate(ind_balanced_list)
        imgs_temp = img_train_ori[img_train_index]
        labels_temp = label_fold[img_train_index]
        # replicate for experts
        abs_imgs = np.tile(imgs_temp, [13, 1, 1, 1])
        labels_temp = np.ndarray.flatten(labels_temp, order='F')
        # get same labels as rsd: 1:plus, 2:prep, 3:normal
        ind_plus = np.where(labels_temp == 2)[0]
        ind_prep = np.where(labels_temp == 1)[0]
        ind_norm = np.where(labels_temp == 0)[0]
        labels_temp[ind_plus] = 1
        labels_temp[ind_prep] = 2
        labels_temp[ind_norm] = 3
        # find categorical labels
        abs_labels, _ = self.biLabels(np.floor(labels_temp))
        ######################################
        # find corresponding comparison labels
        pair_indices = np.arange(0, 5941)
        cmp_indices = self.obtain_comparison_indices(img_train_index, img_names, pair_indices, label_cmp)
        comp_imgs_1, comp_imgs_2, comp_labels = self.load_cmp_data(label_cmp, cmp_indices)
        return abs_imgs, abs_labels, comp_imgs_1, comp_imgs_2, comp_labels

    def load_testing_data(self, kthFold, abs_thr='plus', test_set='100'):
        '''
        loaded labels: 0:normal, 1: prep, 2:plus!
        '''
        # setup for absolute labels
        part_abs_test = self.partition_file_100['RSDTestPlusPartition']
        label_abs = self.partition_file_100['label13']
        img_names = self.partition_file_100['orderName']
        ind_abs_test = part_abs_test[kthFold].astype(np.int)
        # load test images for kthFold
        img_test_list = [self.img_folder_100 + img_names[int(order + 1)] + '.png' for order in ind_abs_test]
        img_test_ori = img_to_array(load_img(img_test_list[0])).astype(np.uint8)[np.newaxis, :, :, :]
        for img_name_iter in img_test_list[1:]:
            img_iter = img_to_array(load_img(img_name_iter)).astype(np.uint8)[np.newaxis, :, :, :]
            img_test_ori = np.concatenate((img_test_ori, img_iter), axis=0)
        # Replicate for all experts
        abs_imgs = np.tile(img_test_ori, [13, 1, 1, 1])
        # Labels
        # replicate for all experts
        label_temp = np.floor(np.ndarray.flatten(label_abs[ind_abs_test], order='F'))
        # choose binary absolute labels
        abs_labels_plus = np.zeros((label_temp.shape[0],))
        abs_labels_plus[np.where(label_temp == 2)[0]] = 1
        abs_labels_prep = 1. * abs_labels_plus
        abs_labels_prep[np.where(label_temp == 1)[0]] = 1
        if abs_thr == 'plus':
            abs_labels = abs_labels_plus
        else:
            abs_labels = abs_labels_prep
        # setup for comparison labels
        part_cmp_test = self.partition_file_100['cmpTestPlusPartition']  # (fold,pair index)
        label_cmp = self.partition_file_100['cmpData']  # (expert,imagename1,imagename2,label)
        # find corresponding comparison labels
        pair_indices = part_cmp_test[kthFold].astype(np.int)
        cmp_indices = self.obtain_comparison_indices(ind_abs_test, img_names, pair_indices, label_cmp)
        comp_imgs_1, comp_imgs_2, comp_labels = self.load_cmp_data(label_cmp, cmp_indices)
        return abs_imgs, abs_labels, comp_imgs_1, comp_imgs_2, comp_labels