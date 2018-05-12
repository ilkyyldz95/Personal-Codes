from __future__ import absolute_import
from __future__ import print_function
from keras.models import load_model
from googlenet_custom_layers import *
from importData import *
from sklearn.metrics import roc_auc_score

'''Imports F and tests with comparison labels'''
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

# LOAD DATA FOR COMPARISON LABELS
kthFold = 0
importer = importData(kthFold)
k_img_test_1, k_img_test_2, k_label_test = importer.importCompTestData()

for data_size in [60000,55000,50000,40000]:
    comp_net = load_model("comp_label_data_size_" + str(data_size) + ".h5",
                          custom_objects={'LRN': LRN, 'PoolHelper': PoolHelper, 'BTPred': BTPred, 'BTLoss': BTLoss})

    # TEST
    y_predict = comp_net.predict([k_img_test_1, k_img_test_2])

    # SAVE RESULTS
    with open('comp_label_test_6000.txt', 'a') as file:
        file.write(str(data_size) + ' training samples.' + '   AUC:' + str(roc_auc_score(k_label_test, y_predict)) + "\n")



