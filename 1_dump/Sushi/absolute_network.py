from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from keras import optimizers
from keras.layers import Dense
from keras.models import Sequential
from sushi_prep import *
from sklearn.metrics import roc_auc_score

class absolute_network(object):
    """
    Training and testing the neural network with absolute labels
    """
    def __init__(self,location='./'):
        self.data_generator = sushi_prep(location=location)

    def train(self, kthFold, save_model_name='./abs_net.h5',
              no_of_labelers=5000, sample_per_user=10, no_of_base_layers=2, max_no_of_nodes=18,
              epochs=100, optimizer=optimizers.RMSprop(), batch_size=128, loss='categorical_crossentropy'):
        """
        Training except k-th fold.
        kthFold: The fold that is going to be tested.
        save_model_name: The file contains the trained model.
        epochs: iterations of training.
        learning_rate:  learning rate for training.
        batch_size: batch size to train.
        loss: Loss function of the nerual network
        """
        features, labels = self.data_generator.sushi_abs_training_data\
                            (kthFold, no_of_labelers=no_of_labelers, sample_per_user=sample_per_user)
        # create base network
        base_net = self.data_generator.create_base_network(no_of_layers=no_of_base_layers, max_no_of_nodes=max_no_of_nodes)
        # add softmax layer
        base_net.add(Dense(1, activation='sigmoid'))
        # train
        base_net.compile(loss=loss, optimizer=optimizer)
        base_net.fit(features, labels, batch_size= batch_size, epochs=epochs)
        base_net.save(save_model_name)

    def test(self, kthFold, model_file, no_of_labelers=5000, sample_per_user=10,
             no_of_base_layers=2, max_no_of_nodes=18, optimizer = optimizers.RMSprop(),
             loss = 'categorical_crossentropy', epochs=100):
        """
        Validating or testing for kthFold images.
        kthFold: int, in [0,4]
        model_file: string, path of the trained model.
        """
        features, labels = self.data_generator.sushi_abs_validation_data \
                        (kthFold, no_of_labelers=no_of_labelers, sample_per_user=sample_per_user)
        # create base network
        base_net = self.data_generator.create_base_network(no_of_layers=no_of_base_layers, max_no_of_nodes=max_no_of_nodes)
        # add softmax layer
        base_net.add(Dense(1, activation='sigmoid'))
        # load model
        base_net.load_weights(model_file)
        base_net.compile(loss=loss, optimizer=optimizer)
        # display results
        predictions = base_net.predict(features)
        with open('test_abs.txt', 'a') as file:
            file.write('\n\nNumber of labelers: ' + str(no_of_labelers))
            file.write('\nSample per user: ' + str(sample_per_user))
            file.write('\nAUC: ' + str(roc_auc_score(labels, predictions)))

    # def compute_abs_accuracy(self, labels, predictions):
    #     '''Compute classification accuracy wrt absolute labels
    #     Predictions are vectors with values between 0 and 1
    #     Labels are categorical
    #     First index is the sample
    #     '''
    #     trues = 0
    #     for i in range(labels.shape[0]):  # for each data sample
    #         currentSample = predictions[i][:].tolist()
    #         currentLabel = labels[i][:].tolist()
    #         trues = trues + (currentSample.index(max(currentSample)) == currentLabel.index(max(currentLabel)))
    #     return 1. * trues / len(labels)
