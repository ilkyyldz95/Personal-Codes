from __future__ import absolute_import
from __future__ import print_function
import sys
from keras.layers import Input, Activation, merge
from keras.models import Model, load_model
from createSigLayer import SigLayer
from createGausLayer import GausLayer
from createInvSigLayer import InvSigLayer
import numpy as np
from googlenet import create_googlenet
import tensorflow as tf
from keras import backend as K
import pickle
from keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import roc_auc_score


'''Trains 3 different G  with absolute labels'''
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
epochs = 50
batch_size = 100     #1 for validation, 32 for train
input_shape = (3,224,224)
loss = absLoss
optimizer = 'sgd'

# LOAD DATA FOR ABSOLUTE LABELS
partition_file = pickle.load(open('./Partitions.p', 'rb'))
img_folder = './preprocessed/All/'
part_rsd_train = partition_file['RSDTrainPlusPartition']
part_rsd_test = partition_file['RSDTestPlusPartition']
label_absolute = partition_file['label13']
label_absolute[label_absolute==1.5] = 2
order_name = partition_file['orderName']
# kthFold = int(sys.argv[1])
kthFold = int(0)
for k in [kthFold]:
    k_ind_rsd_train = part_rsd_train[k]
    k_ind_rsd_test = part_rsd_test[k]
    k_img_train_list = [img_folder+order_name[order+1]+'.png' for order in k_ind_rsd_train]
    k_img_test_list = [img_folder+order_name[order+1]+'.png' for order in k_ind_rsd_test]
# Load Images
    # Image for training
    # k_img_train  = img_to_array(load_img(k_img_train_list[0]))[np.newaxis,:,:,:]
    K_img_test = img_to_array(load_img(k_img_test_list[0]))[np.newaxis,:,:,:]
    for img_name_iter in k_img_test_list[1:]:
        img_iter = img_to_array(load_img(img_name_iter))[np.newaxis,:,:,:]
        K_img_test = np.concatenate((K_img_test,img_iter),axis=0)
    k_img_test = np.tile(K_img_test, [13, 1, 1, 1])
    k_label_abs_test = label_absolute[k_ind_rsd_test,:]
    k_label_test_ori = np.reshape(k_label_abs_test,(-1,),order='F')

    F = create_googlenet(no_classes=hid_layer_dim, no_features=no_of_features)

    # CREATE F&G
    concat_abs_net = create_network(F, hid_layer_dim, input_shape)
    concat_abs_net.load_weights("abs_label_F_save_model_32_F_inputAll_" + str(kthFold) + ".h5")
    # Train all models with corresponding images
    concat_abs_net.compile(loss=loss, optimizer=optimizer)
    sess=K.get_session()
    # for i in range(5):
    #     score_layer_model = Model(inputs=concat_abs_net.input,outputs=concat_abs_net.get_layer(index=i).output)
    #
    # # layer name "prob" index=94
    #     score_predict = score_layer_model.predict(k_img_test)
    #     output0 = score_predict[0]
    #     output1 = score_predict[1]
    #     output5 = score_predict[5]
    #     if np.array_equal(output0,output5):
    #         print("get same output at layer",i)
    #         break
    score_layer_model = Model(inputs=concat_abs_net.input, outputs=concat_abs_net.get_layer(index=94).output)
    score_predict = score_layer_model.predict(k_img_test)

    k_label_test = 1.* k_label_test_ori
    k_label_test[k_label_test==1]=0
    k_label_test[k_label_test==2]=1
    print(k_label_abs_test.shape)
    print(score_predict.shape)
    print(roc_auc_score(k_label_test,score_predict))

    print("Done")
