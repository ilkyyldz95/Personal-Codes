from __future__ import absolute_import
from __future__ import print_function
from keras.layers import Input, Lambda
from keras.models import Model
from keras import optimizers
from googlenet import create_googlenet
from importData import *
from sklearn.metrics import roc_auc_score
from keras import backend as K

'''Imports model and tests with all labels'''
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
    return (1 - alpha) * K.log(1 + exponent)

# INITIALIZE PARAMETERS
kthFold = 0
data_size_abs = 6000
#data_size_comp = 14000
lr=1e-06
sgd = optimizers.SGD(lr=lr)
no_of_features = 1024

# LOAD DATA
importer = importData(kthFold)
k_img_test_abs, k_label_test_abs = importer.importAbsTest6000Data()
k_img_test_1, k_img_test_2, k_label_test_comp = importer.importCompTestData()
# Binary AUC test, Normal(1) vs. Not Normal(0)
k_label_test_abs[k_label_test_abs == 1] = 0
k_label_test_abs[k_label_test_abs == 2] = 1

# LOAD JAMES' NETWORK FOR F
F1 = create_googlenet(no_classes=1, no_features=no_of_features)
F2 = create_googlenet(no_classes=1, no_features=no_of_features)

# do not repeat layer names
for i in range(len(F1.layers)):
    F2.layers[i].name += '_'

# CREATE TWIN NETWORKS: Siamese
input_a = F1.input
input_b = F2.input
score_a = F1.output
score_b = F2.output
distance = Lambda(BTPred, output_shape=(1,))([score_a, score_b])
comp_net = Model([input_a, input_b], distance)


for alpha in [0.0,0.3,0.5,0.7,1.0]:
    # Compile model for current alpha
    comp_net.compile(loss=BTLoss, optimizer=sgd)
    for data_size_comp in [7000, 14000]:
        # Load model
        comp_net.load_weights("abs_comp_train_compnet_" + str(data_size_comp) + '_' + str(alpha) + ".h5")

        ###########################
        # Test for absolute labels
        score_layer_model = Model(inputs=comp_net.input[0], outputs=comp_net.get_layer('prob_modified').output)
        score_predict = score_layer_model.predict(k_img_test_abs)

        with open('abs_comp_label_test_6000.txt', 'a') as file:
            file.write('\nAbsolute AUC for alpha:' + str(alpha) + " and comp data size:" + str(data_size_comp) +
                       '.\tAUC:' + str(roc_auc_score(k_label_test_abs, score_predict))+'\n')

        ##########################
        # Test for comparison labels
        y_predict = comp_net.predict([k_img_test_1, k_img_test_2])

        with open('abs_comp_label_test_6000.txt', 'a') as file:
            file.write('Comparison AUC for alpha:' + str(alpha) + " and comp data size:" + str(data_size_comp) +
                       '.\tAUC:' + str(roc_auc_score(k_label_test_comp, y_predict))+'\n')





