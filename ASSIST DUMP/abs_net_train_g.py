from __future__ import absolute_import
from __future__ import print_function
import sys
from keras.layers import Input, Activation, merge
from keras.models import Model, load_model
from createSigLayer import SigLayer
from createGausLayer import GausLayer
from createInvSigLayer import InvSigLayer
import numpy as np
from googlenet import create_googlenet
from keras import backend as K
from keras import optimizers
from importData import *

'''Trains 3 different G  with absolute labels'''
def biLabels(labels):
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

def create_network(F, hid_layer_dim, input_shape):
    ''' F&G concatenated
    3 parallel networks, with the same F and different Gs'''

    # shared_input = Input(shape=input_shape)
    shared_input = F.input
    # Pass input through F
    # f_out = F(shared_input)
    f_out = F.output

    # Create G
    processed_norm = InvSigLayer(hid_layer_dim)(f_out)
    processed_pre = GausLayer(hid_layer_dim)(f_out)
    processed_plus = SigLayer(hid_layer_dim)(f_out)

    # Add 3 dimensional softmax output
    merge_out = merge([processed_norm, processed_pre, processed_plus], mode='concat')
    activ_out = Activation('softmax')(merge_out)

    # Create the whole network
    concat_abs_net = Model(shared_input, activ_out)
    return concat_abs_net

def absLoss(y_true, y_pred):
    """
    Negative log likelihood of absolute model
    y_true = [100],[010],[001]
    y_pred = soft[g_1(s), g_2(s), g_3(s)]
    Take the g output corresponding to the label
    """
    diff = K.dot(y_pred, K.transpose(y_true))
    return -K.log(diff)

# INITIALIZE PARAMETERS
hid_layer_dim = 1 #F has 1 output: score
no_of_features = 1024
epochs = 25
batch_size = 32     #1 for validation, 100 for prediction
input_shape = (3,224,224)
loss = absLoss
sgd = optimizers.SGD(lr=1e-10)

# LOAD DATA FOR ABSOLUTE LABELS
kthFold = int(sys.argv[1])
# kthFold = int(0)
importer = importData(kthFold)
k_img_train, k_label_train = importer.importAbsTrainData()

# LOAD JAMES' NETWORK FOR F along with ImageNet weights
F_prev = create_googlenet(no_classes=1000, no_features=no_of_features)
F_prev.load_weights("googlenet_weights.h5", by_name=True)
F = create_googlenet(no_classes=hid_layer_dim, no_features=no_of_features)
for i in range(len(F.layers) - 2): #last 2 layers depends on the number of classes
    F.layers[i].set_weights(F_prev.layers[i].get_weights())

# CREATE F&G
concat_abs_net = create_network(F, hid_layer_dim, input_shape)

# Train all models with corresponding images
concat_abs_net.compile(loss=loss, optimizer=sgd)
concat_abs_net.fit(k_img_train, k_label_train, batch_size=batch_size, epochs=epochs)

# Save weights for F
concat_abs_net.save("abs_label_1e-10_" + str(kthFold) + ".h5")