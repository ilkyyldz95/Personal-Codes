from __future__ import absolute_import
from __future__ import print_function
from keras.layers import Input, Activation, merge, Lambda
from keras.models import Model, load_model
from createSigLayer import SigLayer
from createGausLayer import GausLayer
from createInvSigLayer import InvSigLayer
from googlenet import create_googlenet
from keras import backend as K
from importData import *
from keras import optimizers

'''Trains whole network with absolute and comparison labels simultaneously'''
def create_abs_network(F):
    ''' F&G concatenated
    3 parallel networks, with the same F and different Gs'''

    shared_input = F.input
    # Pass input through F
    f_out = F.output

    # Create G
    processed_norm = InvSigLayer()(f_out)
    processed_pre = GausLayer()(f_out)
    processed_plus = SigLayer()(f_out)

    # Add 3 dimensional softmax output
    merge_out = merge([processed_norm, processed_pre, processed_plus], mode='concat')
    activ_out = Activation('softmax')(merge_out)

    # Create the whole network
    concat_abs_net = Model(shared_input, activ_out)
    return concat_abs_net

def BTPred(scalars):
    """
    This is the output when we predict comparison labels. s1, s2 = scalars (beta.*x)
    """
    s1 = scalars[0]
    s2 = scalars[1]
    return s1 - s2

def absLoss(y_true, y_pred):
    """
    Negative log likelihood of absolute model
    y_true = [100],[010],[001]
    y_pred = soft[g_1(s), g_2(s), g_3(s)]
    Take the g output corresponding to the label
    """
    diff = K.dot(y_pred, K.transpose(y_true))
    return -alpha * K.log(diff)

def BTLoss(y_true, y_pred):
    """
    Negative log likelihood of Bradley-Terry Penalty, to be minimized. y = beta.*x
    """
    exponent = K.exp(-y_true * (y_pred))
    return (1-alpha) * K.log(1 + exponent)

# INITIALIZE PARAMETERS
epochs = 3
alpha = float(sys.argv[2]) # balance between absolute and comparison contributions
#alpha = float(0.5) # balance between absolute and comparison contributions
kthFold = int(sys.argv[1])
#kthFold = int(0)

lr=1e-06
sgd = optimizers.SGD(lr=lr)
no_of_features = 1024
batch_size = 32

# LOAD DATA
importer = importData(kthFold)
k_img_train_abs, k_label_train_abs = importer.importAbsTrainData()
k_img_train_1, k_img_train_2, k_label_train_comp = importer.importCompTrainData()

# LOAD JAMES' NETWORK FOR F along with ImageNet weights
F_prev = create_googlenet(no_classes=1000, no_features=no_of_features)
F_prev.load_weights("googlenet_weights.h5", by_name=True)
F1 = create_googlenet(no_classes=1, no_features=no_of_features)
F2 = create_googlenet(no_classes=1, no_features=no_of_features)
for i in range(len(F1.layers) - 2): #last 2 layers depends on the number of classes
    F1.layers[i].set_weights(F_prev.layers[i].get_weights())
    F2.layers[i].set_weights(F_prev.layers[i].get_weights())

# CREATE F&G for both branches
abs_net_1 = create_abs_network(F1)
abs_net_2 = create_abs_network(F2)
abs_net_1.compile(optimizer=sgd, loss=absLoss)
abs_net_2.compile(optimizer=sgd, loss=absLoss)

# do not repeat layer names
for i in range(len(abs_net_2.layers)):
    abs_net_2.layers[i].name += '_'

# CREATE TWIN NETWORKS: Siamese
input_a = F1.input
input_b = F2.input
score_a = F1.output
score_b = F2.output

distance = Lambda(BTPred, output_shape=(1,))([score_a, score_b])
comp_net = Model([input_a, input_b], distance)
comp_net.compile(optimizer=sgd, loss=BTLoss)

# Train: Iteratively train each model at each epoch, with weight of alpha
for epoch in range(epochs):
    abs_net_1.fit(k_img_train_abs, k_label_train_abs, batch_size=batch_size, epochs=1)

    for i in range(len(abs_net_1.layers)):  # last 2 layers depends on the number of classes
        abs_net_2.layers[i].set_weights(abs_net_1.layers[i].get_weights())

    comp_net.fit([k_img_train_1, k_img_train_2], k_label_train_comp, batch_size=batch_size, epochs=1)

    print('*********End of epoch '+str(epoch)+'\n')

# Save weights for F
abs_net_1.save("abs_comp_train_absnet_" + str(kthFold) + '_' + str(alpha) + ".h5")
comp_net.save("abs_comp_train_compnet_" + str(kthFold) + '_' + str(alpha) + ".h5")