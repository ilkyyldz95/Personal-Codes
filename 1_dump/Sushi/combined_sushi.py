from __future__ import absolute_import
from __future__ import print_function
import keras.backend as K
import numpy as np
from math import sqrt
import tensorflow as tf
from keras import optimizers
from keras.layers import Dense, Input, Lambda
from keras.models import Sequential, Model
from sushi_prep import *
from sklearn.metrics import roc_auc_score

def BTPred(scalars):
    """
    This is the output when we predict comparison labels. s1, s2 = scalars (beta.*x)
    """
    s1 = scalars[0]
    s2 = scalars[1]
    return s1 - s2

def scaledBTLoss(no_of_comp):
    def BTLoss(y_true, y_pred):
        """
        Negative log likelihood of Bradley-Terry Penalty, to be minimized. y = beta.*x
        y_true:-1 or 1
        y_pred:si-sj
        """
        exponent = K.exp(-y_true * (y_pred))
        return K.log(1 + exponent) / no_of_comp
    return BTLoss

def scaledCrossEntropy(no_of_abs):
    def crossEntropy(y_true, y_pred):
        return K.binary_crossentropy(y_true, y_pred) / no_of_abs
    return crossEntropy

class combined_network(object):
    """
    Training and testing the neural network with both absolute and comparison labels
    """
    def __init__(self,location='./'):
        self.data_generator = sushi_prep(location=location)

    def train(self, kthFold, save_model_name='./combined_net.h5', train_set = 'both',
              no_of_labelers = 5000, sample_per_user=10, user_sim_thr=0, no_of_base_layers=2, max_no_of_nodes=18,
              epochs=100, optimizer=optimizers.RMSprop(), batch_size=128,
              comp_loss=scaledBTLoss, abs_loss=scaledCrossEntropy):
        """
        Training except k-th fold.
        kthFold: The fold that is going to be tested.
        save_model_name: The file contains the trained model.
        epochs: iterations of training.
        learning_rate:  learning rate for training.
        batch_size: batch size to train.
        loss: Loss function of the neural network
        """
        features_1, features_2, comp_labels, features, abs_labels = self.data_generator.sushi_combined_training_data\
            (kthFold, no_of_labelers=no_of_labelers, sample_per_user=sample_per_user, user_sim_thr=user_sim_thr)
        input_dim = features_1.shape[1]
        no_of_comp_labels = comp_labels.shape[0]
        no_of_abs_labels = abs_labels.shape[0]
        # create base network
        base_net = self.data_generator.create_base_network(no_of_layers=no_of_base_layers, max_no_of_nodes=max_no_of_nodes)
        # add score layer
        score_out = Dense(1, activation='sigmoid')
        # create siamese
        input_a = Input(shape=(input_dim,))
        input_b = Input(shape=(input_dim,))
        base_out_a = base_net(input_a)
        base_out_b = base_net(input_b)
        # comparison part
        score_out_a = score_out(base_out_a)
        score_out_b = score_out(base_out_b)
        distance = Lambda(BTPred, output_shape=(1,))([score_out_a, score_out_b])
        comp_net = Model([input_a, input_b], distance)
        # absolute part
        abs_net = Model(input_a, score_out_a)
        # compile and train
        comp_net.compile(loss=comp_loss(no_of_comp=no_of_comp_labels), optimizer=optimizer)
        abs_net.compile(loss=abs_loss(no_of_abs=no_of_abs_labels), optimizer=optimizer)
        # train on abs only, comp only or both
        if train_set == 'abs':
            abs_net.fit(features, abs_labels, batch_size=batch_size, epochs=epochs)
            # Save weights
            abs_net.save('Abs_' + save_model_name)
        elif train_set == 'comp':
            comp_net.fit([features_1, features_2], comp_labels, batch_size=batch_size, epochs=epochs)
            # Save weights
            comp_net.save('Comp_' + save_model_name)
        else:
            for epoch in range(epochs):
                abs_net.fit(features, abs_labels, batch_size=batch_size, epochs=1)
                comp_net.fit([features_1, features_2], comp_labels, batch_size=batch_size, epochs=1)
                print('**********End of epoch ' + str(epoch))
            # Save weights
            comp_net.save('Both_' + save_model_name)

    def test(self, kthFold, model_file, train_set = 'both',
              no_of_labelers = 5000, sample_per_user=10, user_sim_thr=0, no_of_base_layers=2, max_no_of_nodes=18,
              optimizer=optimizers.RMSprop(), comp_loss=scaledBTLoss, abs_loss=scaledCrossEntropy):
        """
        Training except k-th fold.
        kthFold: The fold that is going to be tested.
        save_model_name: The file contains the trained model.
        epochs: iterations of training.
        learning_rate:  learning rate for training.
        batch_size: batch size to train.
        loss: Loss function of the neural network
        """
        features_1, features_2, comp_labels, features, abs_labels = self.data_generator.sushi_combined_validation_data\
            (kthFold, no_of_labelers=no_of_labelers, sample_per_user=sample_per_user, user_sim_thr=user_sim_thr)
        input_dim = features_1.shape[1]
        no_of_comp_labels = comp_labels.shape[0]
        no_of_abs_labels = abs_labels.shape[0]
        # create base network
        base_net = self.data_generator.create_base_network(no_of_layers=no_of_base_layers, max_no_of_nodes=max_no_of_nodes)
        # add score layer
        score_out = Dense(1, activation='sigmoid')
        # create siamese
        input_a = Input(shape=(input_dim,))
        input_b = Input(shape=(input_dim,))
        base_out_a = base_net(input_a)
        base_out_b = base_net(input_b)
        # comparison part
        score_out_a = score_out(base_out_a)
        score_out_b = score_out(base_out_b)
        distance = Lambda(BTPred, output_shape=(1,))([score_out_a, score_out_b])
        comp_net = Model([input_a, input_b], distance)
        # absolute part
        abs_net = Model(input_a, score_out_a)
        # load weights and compile: BASE NETWORK HAS THE SAME WEIGHTS
        if train_set == 'abs':
            comp_net.load_weights('Abs_' + model_file, by_name=True)
        elif train_set == 'comp':
            comp_net.load_weights('Comp_' + model_file, by_name=True)
        else:
            comp_net.load_weights('Both_' + model_file, by_name=True)
        comp_net.compile(loss=comp_loss(no_of_comp=no_of_comp_labels), optimizer=optimizer)
        abs_net.compile(loss=abs_loss(no_of_abs=no_of_abs_labels), optimizer=optimizer)
        # display results
        abs_pred = abs_net.predict(features)
        comp_pred = comp_net.predict([features_1, features_2])
        with open('test_combined.txt', 'a') as file:
            file.write('\n\nTraining set: ' + str(train_set))
            file.write('\nSimilarity thr: ' + str(user_sim_thr))
            file.write('\nSamples per user: ' + str(sample_per_user))
            file.write('\nAUC on absolute: ' + str(roc_auc_score(abs_labels, abs_pred)))
            file.write('\nAUC on comparison: ' + str(roc_auc_score(comp_labels, comp_pred)))

