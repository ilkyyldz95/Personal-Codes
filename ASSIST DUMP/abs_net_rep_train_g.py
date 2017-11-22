from __future__ import absolute_import
from __future__ import print_function
from keras.models import load_model, Model
from googlenet_custom_layers import *
from createGausLayer import *
from createInvSigLayer import *
from createSigLayer import *
from importData import *

'''Resumes training 3 different G  with absolute labels'''
def absLoss(y_true, y_pred):
    """
    Negative log likelihood of absolute model
    y_true = [100],[010],[001]
    y_pred = soft[g_1(s), g_2(s), g_3(s)]
    Take the g output corresponding to the label
    """
    diff = K.dot(y_pred, K.transpose(y_true))
    return -K.log(diff)

# Initialization
lr = 1e-08
batch_size = 32
epochs = 60

# LOAD DATA FOR ABSOLUTE LABELS
kthFold = int(sys.argv[1])
# kthFold = int(0)
importer = importData(kthFold)
k_img_train, k_label_train = importer.importAbsTrainData()

# Load model to resume training, along with custom layers
concat_abs_net = load_model("abs_label_" + str(lr) + '_' + str(kthFold) + ".h5",
                                custom_objects={'LRN': LRN, 'PoolHelper': PoolHelper,
                                                'InvSigLayer': InvSigLayer,
                                                'GausLayer': GausLayer,
                                                'SigLayer': SigLayer,
                                                'absLoss': absLoss})

# Train
concat_abs_net.fit(k_img_train, k_label_train, batch_size=batch_size, epochs=epochs)

# Save model
concat_abs_net.save("abs_label_"+ str(lr) + '_' + str(kthFold) + "_2ndRep.h5")