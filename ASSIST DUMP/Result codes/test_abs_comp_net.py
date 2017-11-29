from __future__ import absolute_import
from __future__ import print_function
from keras.layers import Input, Activation, merge, Lambda
from keras.models import Model, load_model
from keras import optimizers
from googlenet import create_googlenet
from googlenet_custom_layers import *
from importData import *
from sklearn.metrics import roc_auc_score
from createSigLayer import SigLayer
from createGausLayer import GausLayer
from createInvSigLayer import InvSigLayer
from keras import backend as K

'''Imports model and tests with all labels'''
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
#alpha = float(sys.argv[1]) # balance between absolute and comparison contributions
#alpha = float(0.5) # balance between absolute and comparison contributions
#kthFold = int(sys.argv[1])
kthFold = int(0)

lr=1e-06
sgd = optimizers.SGD(lr=lr)
no_of_features = 1024

# LOAD DATA
importer = importData(kthFold)
k_img_test_abs, k_label_test_abs = importer.importAbsTestData()
k_img_test_1, k_img_test_2, k_label_test_comp = importer.importCompTestData()
# Binary AUC test!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
k_label_test_abs[k_label_test_abs == 1] = 0
k_label_test_abs[k_label_test_abs == 2] = 1

# LOAD JAMES' NETWORK FOR F along with ImageNet weights
F_prev = create_googlenet(no_classes=1000, no_features=no_of_features)
F_prev.load_weights("googlenet_weights.h5", by_name=True)
F1 = create_googlenet(no_classes=1, no_features=no_of_features)
for i in range(len(F1.layers) - 2): #last 2 layers depends on the number of classes
    F1.layers[i].set_weights(F_prev.layers[i].get_weights())

# CREATE F&G for both branches
abs_net_1 = create_abs_network(F1)

for alpha in [0.0,0.2,0.4,0.6,0.8,1.0]:
    # Load model
    comp_net = load_model("abs_comp_train_compnet_" + str(kthFold) + '_' + str(alpha) + ".h5",
                          custom_objects={'LRN': LRN, 'PoolHelper': PoolHelper, 'BTPred': BTPred, 'BTLoss': BTLoss})

    # Load model
    '''abs_net_1 = load_model("abs_comp_train_absnet_" + str(kthFold) + '_' + str(alpha) + ".h5",
                           custom_objects={'LRN': LRN, 'PoolHelper': PoolHelper, 'absLoss': absLoss,
                                           'InvSigLayer': InvSigLayer, 'GausLayer': GausLayer, 'SigLayer': SigLayer})'''
    abs_net_1.compile(optimizer=sgd, loss=absLoss)
    abs_net_1.load_weights("abs_comp_train_absnet_" + str(kthFold) + '_' + str(alpha) + ".h5")

    ###########################
    # Test for absolute labels
    score_layer_model = Model(inputs=abs_net_1.input, outputs=abs_net_1.get_layer('prob_modified').output)
    score_predict = score_layer_model.predict(k_img_test_abs)

    print('Results for absolute labels\n')
    print(str(alpha) + '\tAUC:' + str(roc_auc_score(k_label_test_abs, score_predict))+'\n')
    with open('abs_comp_label_test.txt', 'a') as file:
        file.write('Absolute AUC for alpha:' + str(alpha) + '\tAUC:' + str(roc_auc_score(k_label_test_abs, score_predict))+'\n')

    ##########################
    # Test for comparison labels
    y_predict = comp_net.predict([k_img_test_1, k_img_test_2])

    print('Results for comparison labels\n')
    print(str(alpha) + '\tAUC:' + str(roc_auc_score(k_label_test_comp, y_predict)) + '\n')
    with open('abs_comp_label_test.txt', 'a') as file:
        file.write('Comparison AUC for alpha:' + str(alpha) + '\tAUC:' + str(roc_auc_score(k_label_test_comp, y_predict))+'\n')






