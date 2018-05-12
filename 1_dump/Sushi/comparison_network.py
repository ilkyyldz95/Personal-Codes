from __future__ import absolute_import
from __future__ import print_function
from math import sqrt
import keras.backend as K
import tensorflow as tf
from keras import optimizers
from keras.layers import Dense, Input, Lambda
from keras.models import Sequential, Model
from sklearn.metrics import roc_auc_score
from sushi_prep import *

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

class comparison_network(object):
    """
    Training and testing the neural network with comparison labels only
    """
    def __init__(self,location='./'):
        self.data_generator = sushi_prep(location=location)

    def train(self, kthFold, save_model_name='./comp_net.h5',
              no_of_labelers=5000, sample_per_user=10, no_of_base_layers=2, max_no_of_nodes=18,
              epochs=100, optimizer=optimizers.RMSprop(), batch_size=128, loss=BTLoss):
        """
        Training except k-th fold.
        kthFold: The fold that is going to be tested.
        save_model_name: The file contains the trained model.
        epochs: iterations of training.
        learning_rate:  learning rate for training.
        batch_size: batch size to train.
        loss: Loss function of the neural network
        """
        features_1, features_2, labels = self.data_generator.sushi_comp_training_data\
                                         (kthFold, no_of_labelers=no_of_labelers, sample_per_user=sample_per_user)
        input_dim = features_1.shape[1]
        # create base network
        base_net = self.data_generator.create_base_network(no_of_layers=no_of_base_layers, max_no_of_nodes=max_no_of_nodes)
        # add score layer
        base_net.add(Dense(1, activation='tanh'))
        # create siamese
        input_a = Input(shape=(input_dim,))
        input_b = Input(shape=(input_dim,))
        processed_a = base_net(input_a)
        processed_b = base_net(input_b)
        distance = Lambda(BTPred, output_shape=(1,))([processed_a, processed_b])
        comp_net = Model([input_a, input_b], distance)
        # train
        comp_net.compile(loss=loss, optimizer=optimizer)
        comp_net.fit([features_1, features_2], labels, batch_size= batch_size, epochs=epochs)
        comp_net.save(save_model_name)

    def test(self, kthFold, model_file, no_of_labelers=5000, sample_per_user=10,
             no_of_base_layers=2, max_no_of_nodes=18, optimizer = optimizers.RMSprop(), loss = BTLoss):
        """
        Validating or testing for kthFold images.
        If kthFold=5, test, not validate
        kthFold: int, in [0,4]
        model_file: string, path of the trained model.
        """
        features_1, features_2, labels = self.data_generator.sushi_comp_validation_data \
                                        (kthFold, no_of_labelers=no_of_labelers, sample_per_user=sample_per_user)
        input_dim = features_1.shape[1]
        # create base network
        base_net = self.data_generator.create_base_network(no_of_layers=no_of_base_layers, max_no_of_nodes=max_no_of_nodes)
        # add score layer
        base_net.add(Dense(1, activation='tanh'))
        # create siamese
        input_a = Input(shape=(input_dim,))
        input_b = Input(shape=(input_dim,))
        processed_a = base_net(input_a)
        processed_b = base_net(input_b)
        distance = Lambda(BTPred, output_shape=(1,))([processed_a, processed_b])
        comp_net = Model([input_a, input_b], distance)
        # load model
        comp_net.load_weights(model_file)
        comp_net.compile(loss=loss, optimizer=optimizer)
        # display results
        predictions = comp_net.predict([features_1,features_2])
        with open('test_comp.txt', 'a') as file:
            file.write('\n\nNumber of labelers: ' + str(no_of_labelers))
            file.write('\nSample per user: ' + str(sample_per_user))
            file.write('\nAUC on comparison: ' + str(roc_auc_score(labels, predictions)))
