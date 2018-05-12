from __future__ import absolute_import
from __future__ import print_function
from keras.layers import Activation, merge
from keras.models import Model
from createSigLayer import SigLayer
from createGausLayer import GausLayer
from createInvSigLayer import InvSigLayer
from googlenet import create_googlenet
from keras import backend as K
from keras import optimizers
from importData import *

'''Trains 3 different G  with absolute labels'''
def create_network(F, hid_layer_dim):
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

# LOAD JAMES' NETWORK FOR F along with ImageNet weights
F = create_googlenet(no_classes=hid_layer_dim, no_features=no_of_features)

# CREATE F&G
concat_abs_net = create_network(F, hid_layer_dim)
concat_abs_net.load_weights("abs_label_1e-06_0_3rdRep.h5")

no_of_w = 0
for layer in concat_abs_net.layers:
    w = layer.trainable_weights
    if len(w) > 0:
        print(str(len(w)) + "\n")
        for i in range(len(w)):
            no_of_w = no_of_w + int(np.sum(w[i].shape))
            print(str(no_of_w) + "\n")

########## 36063 weights

