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

# Initialization
lr = 1e-06

for kthFold in [0,1,2,3,4]:
    # LOAD DATA FOR COMPARISON LABELS
    importer = importData(kthFold)
    k_img_test_1, k_img_test_2, k_label_test = importer.importCompTestData()

    comp_net = load_model("comp_label_" + str(lr) + "_" + str(kthFold) + "_3rdRep.h5",
                          custom_objects={'LRN': LRN, 'PoolHelper': PoolHelper, 'BTPred': BTPred, 'BTLoss': BTLoss})

    # TEST
    y_predict = comp_net.predict([k_img_test_1, k_img_test_2])

    # SAVE RESULTS
    with open('comp_label_test.txt', 'a') as file:
        file.write(str(kthFold) + 'th fold' + '   AUC:' + str(roc_auc_score(k_label_test, y_predict)))

    print(min(y_predict))
    print(max(y_predict))
    print(str(kthFold) + 'th fold' + '   AUC:' + str(roc_auc_score(k_label_test, y_predict)))



