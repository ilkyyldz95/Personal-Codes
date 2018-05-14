from __future__ import absolute_import
from __future__ import print_function
from math import sqrt
import keras.backend as K
import numpy as np
import tensorflow as tf
from importData_fac import *
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import SGD
from googlenet_fac import *

def normPred(features):
    """
    This is the output when we predict comparison labels. s1, s2 = scalars (beta.*x)
    """
    f1 = features[0]
    f2 = features[1]
    return tf.norm(f1, axis=1, keep_dims=True)**2 - tf.norm(f2, axis=1, keep_dims=True)**2

def normLoss(y_true, y_pred):
    '''max(||f_pos||^2 - ||f_neg||^2)'''
    return - y_true * y_pred

class combined_fac(object):
    """
    Training and testing the neural network with 5000 images.
    """
    def __init__(self, input_shape=(3, 224, 224), no_of_classes=2, dir="./IMAGE_QUALITY_DATA"):
        self.data_gen = importData(input_shape=input_shape, dir=dir)
        self.input_shape = input_shape
        self.no_of_classes = no_of_classes
        self.data_dir = dir

    def create_siamese(self, reg_param=0.0002, no_of_fused_features=256):
        input1 = Input(shape=self.input_shape)
        input2 = Input(shape=self.input_shape)
        inputRef = Input(shape=self.input_shape)
        # get features from base network
        feature1, feature2 = create_googlenet(input1, input2, inputRef,
                                              reg_param=reg_param, no_of_fused_features=no_of_fused_features)
        distance = Lambda(normPred, output_shape=(1,))([feature1, feature2])
        comp_net = Model([input1, input2, inputRef], distance)
        return comp_net

    def train(self, save_model_name='./combined.h5',
              reg_param=0.0002, no_of_fused_features=256, learning_rate=1e-4, epochs=100, batch_size=32):
        """
        Training CNN except validation and test folds
        """
        ref_imgs, comp_imgs_1, comp_imgs_2, comp_labels = self.data_gen.load_train_data()
        comp_net = self.create_siamese(reg_param=reg_param, no_of_fused_features=no_of_fused_features)
        # load imagenet weights
        comp_net.load_weights('./googlenet_weights.h5', by_name=True)
        comp_net.compile(loss=normLoss, optimizer=SGD(learning_rate))
        # train on abs only, comp only or both
        comp_net.fit([comp_imgs_1, comp_imgs_2, ref_imgs], comp_labels, batch_size=batch_size, epochs=epochs)
        # Save weights
        comp_net.save(save_model_name)

    def test(self, model_file, reg_param=0.0002, no_of_fused_features=256, learning_rate=1e-4):
        """
        Testing CNN on validation/test fold.
        Predict 0th class, the class with the highest score
        """
        abs_imgs, abs_labels = self.data_gen.load_test_data()
        comp_net = self.create_siamese(reg_param=reg_param, no_of_fused_features=no_of_fused_features)
        # get aesthetic representation
        comp_test_model = Model(inputs=comp_net.input[0], outputs=comp_net.get_layer('base_out').get_output_at(0))
        comp_test_model.load_weights(model_file, by_name=True)
        # compile all models
        comp_test_model.compile(loss=normLoss, optimizer=SGD(learning_rate))
        # predict for all filtered images, 3520 * 1024
        abs_pred = comp_test_model.predict(abs_imgs)
        print(abs_pred.shape)
        no_of_ref_im = int(abs_pred.shape[0] / 22)
        # find and sort feature norm for each filtered image, increasing order
        f_norm_all = np.linalg.norm(abs_pred, axis=1)
        scores_per_ref_im = np.reshape(f_norm_all, (no_of_ref_im, 22))
        score_order_per_ref_im = np.argsort(scores_per_ref_im, axis=1)
        # find positive labels for each filtered image
        labels_per_ref_im = np.reshape(abs_labels, (no_of_ref_im, 22))
        # count true positives
        TP_top1 = 0
        TP_top3 = 0
        TP_top5 = 0
        for row in range(no_of_ref_im):
            pos_ind = list(np.where(labels_per_ref_im[row, :] == 1)[0])
            if score_order_per_ref_im[row, -1] in pos_ind:
                TP_top1 += 1
                TP_top3 += 1
                TP_top5 += 1
            elif score_order_per_ref_im[row, -2] in pos_ind or score_order_per_ref_im[row, -3] in pos_ind:
                TP_top3 += 1
                TP_top5 += 1
            elif score_order_per_ref_im[row, -4] in pos_ind or score_order_per_ref_im[row, -5] in pos_ind:
                TP_top5 += 1
        with open('test.txt', 'a') as file:
            file.write('\n\nLambda: ' + str(reg_param) + ' & Learning rate: ' + str(learning_rate))
            file.write('\nTop-1 accuracy: ' + str(1.0 * TP_top1 / no_of_ref_im))
            file.write('\nTop-3 accuracy: ' + str(1.0 * TP_top3 / no_of_ref_im))
            file.write('\nTop-5 accuracy: ' + str(1.0 * TP_top5 / no_of_ref_im))