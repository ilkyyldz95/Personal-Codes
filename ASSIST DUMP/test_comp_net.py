from __future__ import absolute_import
from __future__ import print_function
from keras.layers import Input, Lambda
from keras import optimizers
from keras import backend as K
from googlenet import *
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

# INITIALIZE PARAMETERS
hid_layer_dim = 1 #score
input_shape = (3,224,224)
no_of_features = 1024
loss = BTLoss

# LOAD DATA FOR COMPARISON LABELS
# kthFold = int(sys.argv[1])
kthFold = int(0)
importer = importData(kthFold)
k_img_test_1, k_img_test_2, k_label_test = importer.importCompTestData()

# LOAD JAMES' NETWORK FOR F
F1 = create_googlenet(no_classes=hid_layer_dim, no_features=no_of_features)
F2 = create_googlenet(no_classes=hid_layer_dim, no_features=no_of_features)

# CREATE TWIN NETWORKS: Siamese
# because we re-use the same instance `base_network`, the weights of the network will be shared across the two branches
input_a = F1.input
input_b = F2.input

processed_a = F1(input_a)
processed_b = F2(input_b)

distance = Lambda(BTPred, output_shape=(1,))([processed_a, processed_b])

comp_net = Model([input_a, input_b], distance)

for lr in [1e-04, 1e-05, 1e-06, 1e-07, 1e-08, 1e-09, 1e-10]:
    sgd = optimizers.SGD(lr=lr)

    comp_net.load_weights("comp_label_" + str(lr) + "_" + str(kthFold) + "_2ndRep.h5")
    comp_net.compile(loss=loss, optimizer=sgd)

    # TEST
    y_predict = comp_net.predict([k_img_test_1, k_img_test_2])

    # SAVE RESULTS
    with open('comp_label_test.txt', 'a') as file:
        file.write(str(lr) + '_AUC:' + str(roc_auc_score(k_label_test,y_predict)) +
                   '_min_estimate:' + str(min(y_predict)) + '_max_estimate:' + str(max(y_predict)))

    print(min(y_predict))
    print(max(y_predict))
    print(str(lr) + '_AUC:' + str(roc_auc_score(k_label_test,y_predict)))



