from __future__ import absolute_import
from __future__ import print_function

import pickle

import numpy as np
from googlenet import create_googlenet
from keras import optimizers
from keras.preprocessing.image import load_img, img_to_array
from preprocessing import augment_data, balance_class_data, biLabels, drop_class_data
from sklearn.metrics import roc_auc_score


class deep_ROP(object):
    """
    Training and testing the neural network with 5000 images.
    """
    def __init__(self,partition_file_path='./6000Partitions.p', img_folder_path='./preprocessed_JamesCode/'):
        """
        Find the cross validation partition_file_path and image folder path.
        partition_file_path: 6000.p contains image names, cross-validation splits, and labels.
        img_folder_path: it contains  5000 images with associated names in partition_file.
        """
        self.img_folder_path = img_folder_path
        self.partition_file = pickle.load(open(partition_file_path, 'rb'))
        self.label_rsd = self.partition_file['RSDLabels']
        self.img_names = self.partition_file['imgNames']


    def load_training_data(self, kthFold, balance=True, drop_rate = 1.0,rotation=True, flip=True):
        """
        Load training images and labels for k-th fold with augmentation.
        drop_rate: float, [0.0,1.0], the percentage that kept in training. 1.0 means keep all balanced data.
        rotation: Boolean parameter indicates whether adding rotated images to the training set.
        flip: Boolean parameter indicates whether adding flipped images to the training set.
        """
        part_rsd_train = self.partition_file['RSDTrainPartition']
        # load training images for kthFold.
        ind_rsd_train = part_rsd_train[kthFold].astype(np.int)
        img_train_list = [self.img_folder_path + self.img_names[int(order)] + '.png' for order in ind_rsd_train]
        img_train_ori = img_to_array(load_img(img_train_list[0]))[np.newaxis, :, :, :]
        for img_name_iter in img_train_list[1:]:
            img_iter = img_to_array(load_img(img_name_iter))[np.newaxis, :, :, :]
            img_train_ori = np.concatenate((img_train_ori, img_iter), axis=0)
        # Load training labels for kthFold
        label_rsd_train = self.label_rsd[ind_rsd_train]
        label_train_ori = np.reshape(label_rsd_train, (-1,), order='F')
        label_train_bi, numOfClass = biLabels(np.array(label_train_ori))
        img_train_augmented, label_train_augmented = augment_data(img_train_ori,label_train_bi)

        if not balance:
            # Returning the unbalanced data.
            training_imgs = img_train_augmented
            training_labels = label_train_augmented
        else:
            if drop_rate == 1.0:
                img_train_aug_bal, label_train_aug_bal = balance_class_data(img_train_augmented, label_train_augmented)
            else:
                img_train_aug_bal, label_train_aug_bal = drop_class_data(img_train_augmented, label_train_augmented,drop_rate)
            training_imgs = img_train_aug_bal
            training_labels = label_train_aug_bal
        return training_imgs, training_labels


    def train(self, kthFold, init_weight='./googlenet_weights.h5', save_model_name='./deep_rop.h5',
              epochs=100, learning_rate=1e-4, drop_rate=1.0,batch_size=32, loss='categorical_crossentropy',
              num_of_classes = 3, num_of_features = 1024):
        """
        Training CNN except k-th fold.

        kthFold: The fold that is going to be tested.
        save_model_name: The file contains the trained model.
        epochs: iterations of training.
        learning_rate:  learning rate for training.
        batch_size: batch size to train.
        loss: Loss function of the nerual network
        return:
        """
        training_imgs, training_labels = self.load_training_data(kthFold,drop_rate=drop_rate)
        deep_ROP_net = create_googlenet(no_classes=num_of_classes, no_features=num_of_features)
        deep_ROP_net.load_weights(init_weight, by_name=True)
        deep_ROP_net.compile(loss=loss, optimizer=optimizers.SGD(lr=learning_rate))
        deep_ROP_net.fit(training_imgs, training_labels,batch_size= batch_size, epochs=epochs)
        deep_ROP_net.save(save_model_name)

    def load_testing_data(self,kthFold):
        """
        Load testing images for k-th fold.
        kthFold: int, in [0,4]
        :return:
        img_test: image array: (N, 3, x, y)
        rsd_plus: label for plus category.
        rsd_prep: label for pre-plus or higher categories.
        """
        part_rsd_test = self.partition_file['RSDTestPartition']
        rsd_labels_plus = self.partition_file['RSDLabelsPlus']
        rsd_labels_prep = self.partition_file['RSDLabelsPreP']
        ind_rsd_test = part_rsd_test[kthFold].astype(np.int)
        rsd_plus = rsd_labels_plus[ind_rsd_test]
        rsd_prep = rsd_labels_prep[ind_rsd_test]
        img_test_list = [self.img_folder_path + self.img_names[int(order)] + '.png' for order in ind_rsd_test]
        img_test = img_to_array(load_img(img_test_list[0]))[np.newaxis, :, :, :]
        for img_name_iter in img_test_list[1:]:
            img_iter = img_to_array(load_img(img_name_iter))[np.newaxis, :, :, :]
            img_test = np.concatenate((img_test, img_iter), axis=0)
        return img_test, rsd_plus, rsd_prep

    def test(self,kthFold, model_file, learning_rate=1e-4, loss='categorical_crossentropy',
              num_of_classes = 3, num_of_features = 1024):
        """
        Testing CNN for kthFold images.

        kthFold: int, in [0,4]
        model_file: string, path of the trained model.
        """
        img_test, rsd_plus, rsd_prep = self.load_testing_data(kthFold)
        deep_ROP_net = create_googlenet(no_classes=num_of_classes, no_features=num_of_features)
        deep_ROP_net.load_weights(model_file)
        deep_ROP_net.compile(loss=loss, optimizer=optimizers.SGD(lr=learning_rate))
        if num_of_classes == 1:
            score = deep_ROP_net.predict(img_test)
            print('Plus', roc_auc_score(rsd_plus, score))
            print('Preplus', roc_auc_score(rsd_prep, score))
        else:
            score_3D = deep_ROP_net.predict(img_test)
            score_normal = score_3D[:,2]
            score_preplus = score_3D[:,1]
            score_plus = score_3D[:,0]
            aucPlus = roc_auc_score(rsd_plus,score_plus)
            aucPreP = roc_auc_score(rsd_prep,-score_normal)
            print('Plus-p',roc_auc_score(rsd_plus,score_plus))
            print('Plus-pre', roc_auc_score(rsd_plus, score_preplus))
            print('Plus-no', roc_auc_score(rsd_plus, -score_normal))
            print('Preplus-p', roc_auc_score(rsd_prep, score_plus))
            print('Preplus-pre', roc_auc_score(rsd_prep, score_preplus))
            print('Preplus-no', roc_auc_score(rsd_prep, -score_normal))
