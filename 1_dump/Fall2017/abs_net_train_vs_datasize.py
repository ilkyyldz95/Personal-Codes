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
def create_network(F):
    ''' F&G concatenated
    3 parallel networks, with the same F and different Gs'''
    shared_input = F.input
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
data_size = int(sys.argv[1])

epochs = 30
lr = 1e-06
no_of_features = 1024
batch_size = 32
loss = absLoss
sgd = optimizers.SGD(lr=lr)

# LOAD DATA FOR ABSOLUTE LABELS
kthFold = int(0)
importer = importData(kthFold)
k_img_train, k_label_train = importer.importAbsTrain6000Data(data_size)

# LOAD JAMES' NETWORK FOR F along with ImageNet weights
F_prev = create_googlenet(no_classes=1000, no_features=no_of_features)
F_prev.load_weights("googlenet_weights.h5", by_name=True)
F = create_googlenet(no_classes=1, no_features=no_of_features)
for i in range(len(F.layers) - 2): #last 2 layers depends on the number of classes
    F.layers[i].set_weights(F_prev.layers[i].get_weights())

# CREATE F&G
concat_abs_net = create_network(F)

# Train all models with corresponding images
concat_abs_net.compile(loss=loss, optimizer=sgd)
concat_abs_net.fit(k_img_train, k_label_train, batch_size=batch_size, epochs=epochs)

# Save model
concat_abs_net.save("abs_label_data_size_" + str(data_size) + "_upd.h5")