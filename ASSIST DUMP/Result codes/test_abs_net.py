from __future__ import absolute_import
from __future__ import print_function
from keras.models import Model, load_model
from importData import *
from googlenet_custom_layers import *
from createGausLayer import *
from createInvSigLayer import *
from createSigLayer import *
from sklearn.metrics import roc_auc_score

'''Tests 3 different G  with absolute labels'''
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
lr = 1e-06

for kthFold in range(5):
    # LOAD DATA FOR ABSOLUTE LABELS
    importer = importData(kthFold)
    k_img_test, k_label_test = importer.importAbsTestData()

    # Load model
    concat_abs_net = load_model("abs_label_" + str(lr) + '_' + str(kthFold) + "_2ndRep.h5",
                                custom_objects={'LRN': LRN, 'PoolHelper': PoolHelper,
                                                'InvSigLayer': InvSigLayer,
                                                'GausLayer': GausLayer,
                                                'SigLayer': SigLayer,
                                                'absLoss': absLoss})

    # TEST
    score_layer_model = Model(inputs=concat_abs_net.input, outputs=concat_abs_net.get_layer('prob_modified').output)
    score_predict = score_layer_model.predict(k_img_test)
    #concat_abs_net.layers[95].weights[0].eval(K.get_session())

    # Binary AUC test
    k_label_test[k_label_test==1]=0
    k_label_test[k_label_test==2]=1

    # SAVE RESULTS
    '''with open('abs_label_test.txt', 'a') as file:
        file.write(str(kthFold) + 'th fold' + '   AUC:' + str(roc_auc_score(k_label_test,score_predict)) +
                   '   min_estimate:' + str(min(score_predict)) + '   max_estimate:' + str(max(score_predict)))

    print(min(score_predict))
    print(max(score_predict))'''
    print(str(kthFold) + 'th fold' + '   AUC:' + str(roc_auc_score(k_label_test,score_predict)))
