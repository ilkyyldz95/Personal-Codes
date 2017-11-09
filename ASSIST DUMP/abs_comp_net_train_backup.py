from __future__ import absolute_import
from __future__ import print_function
from keras.layers import Input, Lambda
from keras import optimizers
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
no_of_features = 1024
epochs = 100
batch_size = 32 #1 for validation, 100 for prediction
loss = BTLoss
sgd = optimizers.SGD(lr=10e-6)


# LOAD TRAIN DATA FOR COMPARISON LABELS: tr_pairs, tr_y

# LOAD JAMES' NETWORK FOR F
F1 = create_googlenet(no_classes=hid_layer_dim, no_features=no_of_features)
F2 = create_googlenet(no_classes=hid_layer_dim, no_features=no_of_features)
F1.load_weights("abs_label_F_32_F_inputAll_.h5")
F2.load_weights("abs_label_F_32_F_inputAll_.h5")
print("F loaded")

# CREATE TWIN NETWORKS: Siamese
# because we re-use the same instance `base_network`, the weights of the network will be shared across the two branches
input_a = F1.input
input_b = F2.input

processed_a = F1(input_a)
processed_b = F2(input_b)

distance = Lambda(BTPred, output_shape=(1,))([processed_a, processed_b])

abs_comp_net = Model([input_a, input_b], distance)

# train
abs_comp_net.compile(loss=BTLoss, optimizer=sgd)
abs_comp_net.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, batch_size=batch_size, epochs=epochs)

# Save weights for F
abs_comp_net.save("abs_comp_label_F_32_F_inputAll_" + str(kthFold) + ".h5")
print("Saved model to disk")


