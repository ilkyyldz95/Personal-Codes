from __future__ import absolute_import
from __future__ import print_function
from keras.models import Model, load_model
from createSigLayer import SigLayer
from createGausLayer import GausLayer
from createInvSigLayer import InvSigLayer
from googlenet_custom_layers import *
from keras import backend as K
from importData import *

'''Resume training whole network with absolute and comparison labels simultaneously'''
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
epochs = 10
#alpha = float(sys.argv[2]) # balance between absolute and comparison contributions
alpha = float(0.5) # balance between absolute and comparison contributions
#kthFold = int(sys.argv[1])
kthFold = int(0)

lr=1e-08
batch_size = 32

# LOAD DATA
importer = importData(kthFold)
k_img_train_abs, k_label_train_abs = importer.importAbsTrainData()
k_img_train_1, k_img_train_2, k_label_train_comp = importer.importCompTrainData()

# Load model
abs_net_1 = load_model("abs_comp_train_absnet" + str(kthFold) + ".h5",
                      custom_objects={'LRN': LRN, 'PoolHelper': PoolHelper, 'absLoss': absLoss,
                                      'InvSigLayer': InvSigLayer, 'GausLayer': GausLayer,'SigLayer': SigLayer})

abs_net_2 = load_model("abs_comp_train_absnet" + str(kthFold) + ".h5",
                      custom_objects={'LRN': LRN, 'PoolHelper': PoolHelper, 'absLoss': absLoss,
                                      'InvSigLayer': InvSigLayer, 'GausLayer': GausLayer,'SigLayer': SigLayer})

# Load model
comp_net = load_model("abs_comp_train_compnet" + str(kthFold) + ".h5",
                      custom_objects={'LRN': LRN, 'PoolHelper': PoolHelper, 'BTPred': BTPred, 'BTLoss': BTLoss})


# Train: Iteratively train each model at each epoch, with weight of alpha
for epoch in range(epochs):
    abs_net_1.fit(k_img_train_abs, k_label_train_abs, batch_size=batch_size, epochs=1)

    for i in range(len(abs_net_1.layers)):  # last 2 layers depends on the number of classes
        abs_net_2.layers[i].set_weights(abs_net_1.layers[i].get_weights())

    comp_net.fit([k_img_train_1, k_img_train_2], k_label_train_comp, batch_size=batch_size, epochs=1)

# Save weights for F
abs_net_1.save("abs_comp_train_absnet" + str(kthFold) + "_2ndRep.h5")
comp_net.save("abs_comp_train_compnet" + str(kthFold) + "_2ndRep.h5")