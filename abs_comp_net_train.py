from __future__ import absolute_import
from __future__ import print_function
from keras.layers import Input, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
from googlenet import *

'''Imports F trained with absolute labels and trains with comparison labels'''

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
    """
    exponent = K.exp(-y_true * (y_pred))
    return K.log(1 + exponent)

# INITIALIZE PARAMETERS
hid_layer_dim = 1 #score
input_shape = (3,224,224)
no_of_images = 196
no_of_features = 1024
epochs = 10
batch_size = 1
loss = BTLoss

# LOAD TRAIN DATA FOR COMPARISON LABELS: tr_pairs, tr_y

# LOAD JAMES' NETWORK FOR F, call abs_net
abs_net = create_googlenet(no_classes=hid_layer_dim, no_features=no_of_features)
abs_net.load_weights("abs_label_F.h5")
print("F loaded")

# CREATE TWIN NETWORKS
# because we re-use the same instance `base_network`, the weights of the network will be shared across the two branches
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

processed_a = abs_net(input_a)
processed_b = abs_net(input_b)

distance = Lambda(BTPred, output_shape=(1,))([processed_a, processed_b])

abs_comp_net = Model([input_a, input_b], distance)

# train
rms = RMSprop()
abs_comp_net.compile(loss=BTLoss, optimizer=rms)
abs_comp_net.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, batch_size=batch_size, epochs=epochs)


