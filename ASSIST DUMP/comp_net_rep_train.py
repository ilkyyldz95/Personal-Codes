from __future__ import absolute_import
from __future__ import print_function
from keras.models import load_model
from googlenet_custom_layers import *
from importData import *

'''Resumes training with comparison labels'''
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

# Initialization
lr = 1e-06
batch_size = 32
epochs = 18

# LOAD DATA FOR COMPARISON LABELS
kthFold = int(sys.argv[1])
# kthFold = int(0)
importer = importData(kthFold)
k_img_train_1, k_img_train_2, k_label_train = importer.importCompTrainData()

# Load model
comp_net = load_model("comp_label_" + str(lr) + "_" + str(kthFold) + ".h5",
                      custom_objects={'LRN': LRN, 'PoolHelper': PoolHelper, 'BTPred': BTPred, 'BTLoss': BTLoss})

# train
comp_net.fit([k_img_train_1, k_img_train_2], k_label_train, batch_size=batch_size, epochs=epochs)

# Save model
comp_net.save("comp_label_" + str(lr) + "_" + str(kthFold) + "_2ndRep.h5")


