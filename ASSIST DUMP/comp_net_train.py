from __future__ import absolute_import
from __future__ import print_function
from keras.layers import Input, Lambda
from keras import optimizers
from keras import backend as K
from googlenet import *
from importData import *

'''Imports F and trains with comparison labels'''

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
epochs = 13
batch_size = 32 #1 for validation, 100 for prediction
loss = BTLoss
lr = float(sys.argv[1])
sgd = optimizers.SGD(lr=lr)

# LOAD DATA FOR COMPARISON LABELS
# kthFold = int(sys.argv[1])
kthFold = int(0)
importer = importData(kthFold)
k_img_train_1, k_img_train_2, k_label_train = importer.importCompTrainData()

# LOAD JAMES' NETWORK FOR F
F_prev = create_googlenet(no_classes=1000, no_features=no_of_features)
F_prev.load_weights("googlenet_weights.h5", by_name=True)
F1 = create_googlenet(no_classes=hid_layer_dim, no_features=no_of_features)
F2 = create_googlenet(no_classes=hid_layer_dim, no_features=no_of_features)
for i in range(len(F1.layers) - 2): #last 2 layers depends on the number of classes
    F1.layers[i].set_weights(F_prev.layers[i].get_weights())
    F2.layers[i].set_weights(F_prev.layers[i].get_weights())
print("F loaded")

# CREATE TWIN NETWORKS: Siamese
# because we re-use the same instance `base_network`, the weights of the network will be shared across the two branches
input_a = F1.input
input_b = F2.input

processed_a = F1(input_a)
processed_b = F2(input_b)

distance = Lambda(BTPred, output_shape=(1,))([processed_a, processed_b])

comp_net = Model([input_a, input_b], distance)

# train
comp_net.compile(loss=loss, optimizer=sgd)
comp_net.fit([k_img_train_1, k_img_train_2], k_label_train, batch_size=batch_size, epochs=epochs)

# Save weights for F
comp_net.save("comp_label_" + str(lr) + "_" + str(kthFold) + ".h5")


