from __future__ import absolute_import
from __future__ import print_function

import pickle

import numpy as np
from custom_layer_setup import *
from googlenet import create_googlenet
from keras import optimizers
from keras.preprocessing.image import load_img, img_to_array
from preprocessing import augment_data, balance_class_data, biLabels
from sklearn.metrics import roc_auc_score


class deep_ROP(object):
    """
    Training and testing the neural network with 5000 images.
    """
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

    def load_training_data(self, kthFold, balance=True, rotation=True, flip=True, num_unique_images=12):
        """
        Load training images and labels for k-th fold with augmentation.
        drop_rate: float, [0.0,1.0], the percentage that kept in training. 1.0 means keep all balanced data.
        rotation: Boolean parameter indicates whether adding rotated images to the training set.
        flip: Boolean parameter indicates whether adding flipped images to the training set.
        """
        part_rsd_train = self.partition_file_100['RSDTrainPlusPartition']
        label_rsd = self.partition_file_100['RSDLabels']
        img_names = self.partition_file_100['orderName']
        # load training images for kthFold. load all if kthFold is 5.
        if kthFold == 5:
            ind_rsd_train = np.arange(100).astype(np.int)
        else:
            ind_rsd_train = part_rsd_train[kthFold].astype(np.int)
        img_train_list = [self.img_folder_100 + img_names[int(order + 1)] + '.png' for order in ind_rsd_train]
        img_train_ori = img_to_array(load_img(img_train_list[0]))[np.newaxis, :, :, :]
        for img_name_iter in img_train_list[1:]:
            img_iter = img_to_array(load_img(img_name_iter))[np.newaxis, :, :, :]
            img_train_ori = np.concatenate((img_train_ori, img_iter), axis=0)
        # Load training labels for kthFold
        label_rsd_train = label_rsd[ind_rsd_train]
        label_train_ori = np.reshape(label_rsd_train, (-1,), order='F')
        label_train_bi, numOfClass = biLabels(np.array(label_train_ori))
        # Balance the data from original images (not augmented images)
        if not balance:
            c = numOfClass
            ind_list = []
            len_ind_list = []
            for c_iter in range(c):
                ind_list.append(np.where(label_train_bi[:, c_iter] == 1)[0])
                len_ind_list.append(len(ind_list[c_iter]))
            np.random.seed(1)
            ind_balanced_list = []
            # keep the ratio of classes while dropping
            for c_balance_iter in range(c):
                no_sample = int(np.floor(len_ind_list[c_balance_iter] * num_unique_images / sum(len_ind_list)))
                if no_sample < 1:
                    raise ValueError("Enter a higher number")
                else:
                    ind_balanced_list.append(
                    ind_list[c_balance_iter][np.random.choice(len_ind_list[c_balance_iter], no_sample, replace=False)])
            img_train_index = np.concatenate(ind_balanced_list)
            img_train_bal = img_train_ori[img_train_index, :, :, :]
            label_train_bal = label_train_bi[img_train_index, :]
            # augmentation
            training_imgs, training_labels = augment_data(img_train_bal, label_train_bal, rotation, flip)
        else:
            img_train_bal, label_train_bal, img_train_index = balance_class_data(img_train_ori, label_train_bi,
                                                                                 num_unique_images=num_unique_images)
            # augmentation
            training_imgs, training_labels = augment_data(img_train_bal, label_train_bal, rotation, flip)
        return training_imgs, training_labels

    def train(self, kthFold, init_weight='./googlenet_weights.h5', save_model_name='./deepRop.h5',
              epochs=100, learning_rate=1e-4, num_unique_images=12, batch_size=32, loss='categorical_crossentropy',
              num_of_classes=3, num_of_features=1024, balance=True, toggle_custom=False):
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
        training_imgs, training_labels = self.load_training_data(kthFold, balance=balance, num_unique_images=num_unique_images)

        if toggle_custom:  #add custom layers after the score output
            F_prev = create_googlenet(no_classes=1000, no_features=num_of_features)
            F_prev.load_weights(init_weight, by_name=True)
            F = create_googlenet(no_classes=1, no_features=num_of_features)
            for i in range(len(F.layers) - 2):  # last 2 layers depends on the number of classes
                F.layers[i].set_weights(F_prev.layers[i].get_weights())
            deep_ROP_net = add_custom(F)
        else:           #use james network as is with 3 classes
            deep_ROP_net = create_googlenet(no_classes=num_of_classes, no_features=num_of_features)
            deep_ROP_net.load_weights(init_weight, by_name=True)

        deep_ROP_net.compile(loss=loss, optimizer=optimizers.SGD(lr=learning_rate))
        deep_ROP_net.fit(training_imgs, training_labels, batch_size= batch_size, epochs=epochs)
        deep_ROP_net.save(save_model_name)

    def load_testing_data(self,kthFold, test_100=True):
        """
        Load testing images for k-th fold.
        kthFold: int, in [0,4]
        :return:
        img_test: image array: (N, 3, x, y)
        rsd_plus: label for plus category.
        rsd_prep: label for pre-plus or higher categories.
        """
        # decide whether to test with first 100 or all 5000
        if test_100: #validation on kthFold
            part_rsd_test = self.partition_file_100['RSDTestPlusPartition']
            label_rsd = self.partition_file_100['RSDLabels']
            img_names = self.partition_file_100['orderName']
            rsd_labels_plus = -1. * np.ones((label_rsd.shape[0],))
            rsd_labels_plus[np.where(label_rsd == 1)[0]] = 1
            rsd_labels_prep = 1. * rsd_labels_plus
            rsd_labels_prep[np.where(label_rsd == 2)[0]] = 1
            ind_rsd_test = part_rsd_test[kthFold].astype(np.int)
            rsd_plus = rsd_labels_plus[ind_rsd_test]
            rsd_prep = rsd_labels_prep[ind_rsd_test]
            img_test_list = [self.img_folder_100 + img_names[int(order) + 1] + '.png' for order in ind_rsd_test]
            img_test = img_to_array(load_img(img_test_list[0]))[np.newaxis, :, :, :]
            for img_name_iter in img_test_list[1:]:
                img_iter = img_to_array(load_img(img_name_iter))[np.newaxis, :, :, :]
                img_test = np.concatenate((img_test, img_iter), axis=0)
        else:           #test with all 5000
            part_rsd_test = self.partition_file_6000['RSDTestPartition']
            img_names = self.partition_file_6000['imgNames']
            rsd_labels_plus = self.partition_file_6000['RSDLabelsPlus']
            rsd_labels_prep = self.partition_file_6000['RSDLabelsPreP']
            ind_rsd_test = part_rsd_test[0].astype(np.int)
            for k in [1, 2, 3, 4]:
                ind_rsd_test = np.append(ind_rsd_test, part_rsd_test[k].astype(np.int))
            rsd_plus = rsd_labels_plus[ind_rsd_test]
            rsd_prep = rsd_labels_prep[ind_rsd_test]
            img_test_list = [self.img_folder_6000 + img_names[int(order)] + '.png' for order in ind_rsd_test]
            img_test = img_to_array(load_img(img_test_list[0]))[np.newaxis, :, :, :]
            for img_name_iter in img_test_list[1:]:
                img_iter = img_to_array(load_img(img_name_iter))[np.newaxis, :, :, :]
                img_test = np.concatenate((img_test, img_iter), axis=0)

        return img_test, rsd_plus, rsd_prep

    def test(self, kthFold, model_file, learning_rate=1e-4, loss='categorical_crossentropy',
             num_of_classes=3, num_of_features=1024, test_100=True, num_unique_images=12, toggle_custom=False):
        """
        Testing CNN for kthFold images.

        kthFold: int, in [0,4]
        model_file: string, path of the trained model.
        """
        img_test, rsd_plus, rsd_prep = self.load_testing_data(kthFold, test_100)
        if toggle_custom:  #add custom layers after the score output
            #create network
            F = create_googlenet(no_classes=1, no_features=num_of_features)
            deep_ROP_net = add_custom(F)
            #load model
            deep_ROP_net.load_weights(model_file)
            deep_ROP_net.compile(loss=loss, optimizer=optimizers.SGD(lr=learning_rate))
            #get 1D score
            score_layer_model = Model(inputs=deep_ROP_net.input,
                                      outputs=deep_ROP_net.get_layer('prob_modified').output)
            score_3D = deep_ROP_net.predict(img_test)
            score_1D = score_layer_model.predict(img_test)
            score_normal = score_3D[:, 2]
            score_preplus = score_3D[:, 1]
            score_plus = score_3D[:, 0]
            #display results
            print(str(loss))
            #display 3D tests
            print('3D plus', roc_auc_score(rsd_plus, score_plus))
            print('3D normal', roc_auc_score(rsd_prep, -score_normal))
            #display 1D tests
            print('1D plus', roc_auc_score(rsd_plus, score_1D))
            print('1D normal', roc_auc_score(rsd_prep, -score_1D))

        else:           #use james network as is with 3 classes
            deep_ROP_net = create_googlenet(no_classes=num_of_classes, no_features=num_of_features)
            deep_ROP_net.load_weights(model_file)
            deep_ROP_net.compile(loss=loss, optimizer=optimizers.SGD(lr=learning_rate))
            if num_of_classes == 1:
                score = deep_ROP_net.predict(img_test)
                # display results
                print('Plus', roc_auc_score(rsd_plus, score))
                print('Preplus', roc_auc_score(rsd_prep, score))
            else:
                score_3D = deep_ROP_net.predict(img_test)
                score_normal = score_3D[:, 2]
                score_preplus = score_3D[:, 1]
                score_plus = score_3D[:, 0]
                with open('test_rsd.txt', 'a') as file:
                    file.write('\n\nFold: ' + str(kthFold))
                    file.write('\nNumber of unique training images: ' + str(num_unique_images))
                    file.write('\nPlus vs others: ' + str(roc_auc_score(rsd_plus, score_plus)))
                    file.write('\nNormal vs. others: ' + str(roc_auc_score(rsd_prep, -score_normal)))
