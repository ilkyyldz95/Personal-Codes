from __future__ import absolute_import
from __future__ import print_function
import sys
from keras.layers import Input, Activation, merge
from keras.models import Model
from createSigLayer import SigLayer
from createGausLayer import GausLayer
from createInvSigLayer import InvSigLayer
import numpy as np
from googlenet import create_googlenet
from keras import optimizers
from keras import backend as K
from importData import *
from sklearn.metrics import roc_auc_score


'''Tests 3 different G  with absolute labels'''
def biLabels(labels):
    """
    This function will binarize labels.
    There are C classes {1,2,3,4,...,c} in the labels, the output would be c dimensional vector.
    Input:
        - labels: (N,) np array. The element value indicates the class index.
    Output:
        - biLabels: (N, C) array. Each row has and only has a 1, and the other elements are all zeros.
    Example:
        The input labels = np.array([1,2,2,1,3])
        The binaried labels are np.array([[1,0,0],[0,1,0],[0,1,0],[1,0,0],[0,0,1]])
    """
    N = labels.shape[0]
    labels.astype(np.int)
    C = len(np.unique(labels))
    binarized = np.zeros((N, C))
    binarized[np.arange(N).astype(np.int), labels.astype(np.int)] = 1
    return binarized

def create_network(F, hid_layer_dim, input_shape):
    ''' F&G concatenated
    3 parallel networks, with the same F and different Gs'''

    # shared_input = Input(shape=input_shape)
    shared_input = F.input
    # Pass input through F
    # f_out = F(shared_input)
    f_out = F.output

    # Create G
    processed_norm = InvSigLayer(hid_layer_dim)(f_out)
    processed_pre = GausLayer(hid_layer_dim)(f_out)
    processed_plus = SigLayer(hid_layer_dim)(f_out)

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
hid_layer_dim = 1 #F has 1 output: score
no_of_features = 1024
input_shape = (3,224,224)
loss = absLoss
sgd = optimizers.SGD(lr=1e-10)

# LOAD JAMES' NETWORK and load weights
F = create_googlenet(no_classes=hid_layer_dim, no_features=no_of_features)
# CREATE F&G
concat_abs_net = create_network(F, hid_layer_dim, input_shape)

for kthFold in range(5):
    # LOAD DATA FOR ABSOLUTE LABELS
    # kthFold = int(sys.argv[1])
    # kthFold = int(1)
    importer = importData(kthFold)
    k_img_test, k_label_test = importer.importAbsTestData()

    concat_abs_net.load_weights("abs_label_1e-10_" + str(kthFold) + ".h5")
    concat_abs_net.compile(loss=loss, optimizer=sgd)

    # TEST
    score_layer_model = Model(inputs=concat_abs_net.input, outputs=concat_abs_net.get_layer('prob_modified').output)
    score_predict = score_layer_model.predict(k_img_test)
    #concat_abs_net.layers[95].weights[0].eval(K.get_session())

    k_label_test = 1.* k_label_test
    k_label_test[k_label_test==1]=0
    k_label_test[k_label_test==2]=1

    # SAVE RESULTS
    with open('abs_label_test.txt', 'a') as file:
        file.write(str(kthFold) + 'th fold' + '_AUC:' + str(roc_auc_score(k_label_test,score_predict)) +
                   '_min_estimate:' + str(min(score_predict)) + '_max_estimate:' + str(max(score_predict)))

    print(min(score_predict))
    print(max(score_predict))
    print(str(kthFold) + 'th fold' + '_AUC:' + str(roc_auc_score(k_label_test,score_predict)))
