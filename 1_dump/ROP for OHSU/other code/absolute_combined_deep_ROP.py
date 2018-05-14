from __future__ import absolute_import
from __future__ import print_function

import keras.backend as K
from absolute_importData import *
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import SGD
from sklearn.metrics import roc_auc_score, accuracy_score

from ROP.code.googlenet_functional import *


def BTPred(scalars):
    """
    This is the output when we predict comparison labels. s1, s2 = scalars (beta.*x)
    """
    s1 = scalars[0]
    s2 = scalars[1]
    return s1 - s2

def scaledBTLoss(alpha):
    def BTLoss(y_true, y_pred):
        """
        Negative log likelihood of Bradley-Terry Penalty, to be minimized. y = beta.*x
        y_true:-1 or 1
        y_pred:si-sj
        alpha: 0-1
        """
        exponent = K.exp(-y_true * (y_pred))
        return (1-alpha) * K.log(1 + exponent)
    return BTLoss

def scaledCrossEntropy(alpha):
    def crossEntropy(y_true, y_pred):
        return alpha * K.categorical_crossentropy(y_true, y_pred)
    return crossEntropy

class combined_deep_ROP(object):
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
        self.data_gen = importData(partition_file_100=partition_file_100, img_folder_100=img_folder_100,
                                   partition_file_6000=partition_file_6000, img_folder_6000=img_folder_6000)

    def create_siamese(self, input_shape=(3, 224, 224), reg_param=0.0002, no_of_classes=3, no_of_score_layers=1,
                       max_no_of_nodes=128):
        input1 = Input(shape=input_shape)
        input2 = Input(shape=input_shape)
        # get features from base network
        feature1, feature2 = create_googlenet(input1, input2, reg_param=reg_param)
        # create and pass through score layers
        score1 = feature1
        score2 = feature2
        for l in range(no_of_score_layers-1):
            layer = Dense(int(max_no_of_nodes / (l+2)), activation='relu', kernel_regularizer=l2(reg_param), name='s'+str(l))
            score1 = layer(score1)
            score2 = layer(score2)
        # create final layers of absolute and comparison
        abs_out = Dense(no_of_classes, activation='softmax', kernel_regularizer=l2(reg_param), name='abs')
        comp_out = Dense(1, activation='sigmoid', kernel_regularizer=l2(reg_param), name='comp')
        # absolute part
        abs_out1 = abs_out(score1)
        abs_net = Model(input1, abs_out1)
        # comparison part
        comp_out1 = comp_out(score1)
        comp_out2 = comp_out(score2)
        distance = Lambda(BTPred, output_shape=(1,))([comp_out1, comp_out2])
        comp_net = Model([input1, input2], distance)
        return abs_net, comp_net

    def train(self, kthFold, save_model_name='./deepRop.h5',
              reg_param=0.0002, no_of_classes=3, no_of_score_layers=1, max_no_of_nodes=128,
              abs_loss=scaledCrossEntropy, comp_loss=scaledBTLoss, alpha=0.5, learning_rate=1e-4,
              epochs=100, num_unique_images=80, batch_size=32, balance=True):
        """
        Training CNN except k-th fold.
        """
        abs_imgs, abs_labels, comp_imgs_1, comp_imgs_2, comp_labels = \
                        self.data_gen.load_training_data(balance=balance, num_unique_images=num_unique_images)
        abs_net, comp_net = self.create_siamese(reg_param=reg_param, no_of_classes=no_of_classes,
                                                no_of_score_layers=no_of_score_layers, max_no_of_nodes=max_no_of_nodes)
        # load imagenet weights
        abs_net.compile(loss=abs_loss(alpha=alpha), optimizer=SGD(learning_rate))
        abs_net.load_weights('./googlenet_weights.h5', by_name=True)
        comp_net.compile(loss=comp_loss(alpha=alpha), optimizer=SGD(learning_rate))
        # train on abs only, comp only or both
        for epoch in range(epochs):
            abs_net.fit(abs_imgs, abs_labels, batch_size=batch_size, epochs=1)
            comp_net.fit([comp_imgs_1, comp_imgs_2], comp_labels, batch_size=batch_size, epochs=1)
            print('**********End of epoch ' + str(epoch))
        # Save weights
        abs_net.save('Bias_Abs_' + save_model_name)
        comp_net.save('Bias_Comp_' + save_model_name)
        # with open('Hist_Comp_' + save_model_name, 'wb') as file_pi:
        #    pickle.dump(history.history, file_pi)

    def test(self, kthFold, model_file,
              reg_param=0.0002, no_of_classes=3, no_of_score_layers=1, max_no_of_nodes=128,
              abs_loss=scaledCrossEntropy, comp_loss=scaledBTLoss, alpha=0.5, learning_rate=1e-4,
              num_unique_images=80, balance=True, abs_test_thr='plus'):
        """
        Testing CNN except k-th fold.
        """
        abs_imgs, abs_labels, comp_imgs_1, comp_imgs_2, comp_labels = \
                        self.data_gen.load_testing_data(kthFold, abs_thr=abs_test_thr, test_set='100')
        abs_net, comp_net = self.create_siamese(reg_param=reg_param, no_of_classes=no_of_classes,
                                                no_of_score_layers=no_of_score_layers, max_no_of_nodes=max_no_of_nodes)
        # load weights and compile: BASE NETWORK HAS THE SAME WEIGHTS
        # make sure we have weights everytime we test. we have two single input-single output branches
        comp_test_model = Model(inputs=comp_net.input[0], outputs=comp_net.get_layer('comp').get_output_at(0))
        comp_test_model.load_weights('Bias_Comp_' + model_file, by_name=True)
        abs_net.load_weights('Bias_Abs_' + model_file, by_name=True)
        comp_net.load_weights('Bias_Comp_' + model_file, by_name=True)
        # compile all models
        comp_test_model.compile(loss=comp_loss(alpha=alpha), optimizer=SGD(learning_rate))
        abs_net.compile(loss=abs_loss(alpha=alpha), optimizer=SGD(learning_rate))
        comp_net.compile(loss=comp_loss(alpha=alpha), optimizer=SGD(learning_rate))
        #################TEST ON 100
        if alpha == 0.0: # only comparison training, use comp models
            comp_pred = comp_net.predict([comp_imgs_1, comp_imgs_2])
            abs_pred = comp_test_model.predict(abs_imgs)
        elif alpha == 1.0: # only absolute training, use abs models
            abs_pred = abs_net.predict(abs_imgs)[:, 0] # take the plus output, needs scalar
            comp_pred = abs_net.predict(comp_imgs_1)[:, 0] - abs_net.predict(comp_imgs_2)[:, 0]
        else:
            abs_pred = abs_net.predict(abs_imgs)[:, 0]
            comp_pred = comp_net.predict([comp_imgs_1, comp_imgs_2])
        with open('bias_test_auc_fold3.txt', 'a') as file:
            file.write('\n\nNo of unique images: ' + str(num_unique_images))
            file.write('\nAlpha: ' + str(alpha))
            file.write('\nLambda: ' + str(reg_param))
            file.write('\nLearning rate: ' + str(learning_rate))
            file.write('\nTest AUC on absolute: ' + str(roc_auc_score(abs_labels, abs_pred)))
            file.write('\nTest AUC on comparison: ' + str(roc_auc_score(comp_labels, comp_pred)))
        #################TEST ON 100 ACCURACY
        if alpha == 0.0: # only comparison training, use comp models
            abs_pred = comp_test_model.predict(abs_imgs)
            # a scalar output, classify with threshold 0.5
            abs_pred_thresholded = (abs_pred > 0.5).astype(int)
        elif alpha == 1.0: # only absolute training, use abs models
            abs_pred = abs_net.predict(abs_imgs)
            # a 3 class output, take the maximum
            abs_pred_012 = np.argmax(abs_pred, axis=1).astype(int)
            abs_pred_thresholded = (abs_pred_012 == 0).astype(int)
        else:
            abs_pred = abs_net.predict(abs_imgs)
            # a 3 class output, take the maximum
            abs_pred_012 = np.argmax(abs_pred, axis=1).astype(int)
            abs_pred_thresholded = (abs_pred_012 == 0).astype(int)
        with open('bias_test_accuracy_fold3.txt', 'a') as file:
            file.write('\n\nNo of unique images: ' + str(num_unique_images))
            file.write('\nAlpha: ' + str(alpha))
            file.write('\nLambda: ' + str(reg_param))
            file.write('\nLearning rate: ' + str(learning_rate))
            file.write('\nTest accuracy on absolute: ' + str(accuracy_score(abs_labels, abs_pred_thresholded)))


