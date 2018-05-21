from __future__ import absolute_import
from __future__ import print_function
import sys
from keras.layers import Input, Activation, merge
from keras.models import Model, load_model
import numpy as np
from googlenet import create_googlenet
import tensorflow as tf
from keras import backend as K
import pickle
from keras.preprocessing.image import load_img, img_to_array
from scipy.ndimage import rotate
#import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from keras import optimizers

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
    binarized[np.arange(N).astype(np.int), labels.astype(np.int)-1] = 1
    return binarized

# INITIALIZE PARAMETERS
hid_layer_dim = 3  # F has 1 output: score
no_of_features = 1024
epochs = 10
batch_size = 32  # 1 for validation, 100 for prediction
input_shape = (3, 224, 224)
loss = 'categorical_crossentropy'
sgd = optimizers.SGD(lr=10e-6)

# LOAD DATA FOR ABSOLUTE LABELS
partition_file = pickle.load(open('./6000Partitions.p', 'rb'))
img_folder = 'preprocessed_JamesCode/'
part_rsd_train = partition_file['RSDTrainPartition']
part_rsd_test = partition_file['RSDTestPartition']
label_rsd = partition_file['RSDLabels']
img_names = partition_file['imgNames']
rsd_labels_plus =partition_file['RSDLabelsPlus']
rsd_labels_prep = partition_file['RSDLabelsPreP']
#kthFold = int(0)
kthFold = int(sys.argv[1])
for k in [kthFold]:
    k_ind_rsd_train = part_rsd_train[k].astype(np.int)
    k_ind_rsd_test = part_rsd_test[k].astype(np.int)
    k_rsd_plus = rsd_labels_plus[k_ind_rsd_test]
    k_rsd_prep = rsd_labels_prep[k_ind_rsd_test]
    k_img_train_list = [img_folder+img_names[int(order)]+'.png' for order in k_ind_rsd_train]
    k_img_test_list = [img_folder+img_names[int(order)]+'.png' for order in k_ind_rsd_test]

# Load Iamges for testing
#     k_img_train_ori = img_to_array(load_img(k_img_train_list[0]))[np.newaxis, :, :, :]
#     for img_name_iter in k_img_train_list[1:]:
#         img_iter = img_to_array(load_img(img_name_iter))[np.newaxis,:,:,:]
#         k_img_train_ori = np.concatenate((k_img_train_ori,img_iter),axis=0)
#     k_label_abs_train = label_rsd[k_ind_rsd_train]

    k_img_test = img_to_array(load_img(k_img_test_list[0]))[np.newaxis, :, :, :]
    for img_name_iter in k_img_test_list[1:]:
        img_iter = img_to_array(load_img(img_name_iter))[np.newaxis, :, :, :]
        k_img_test = np.concatenate((k_img_test,img_iter),axis=0)

    concat_abs_net = create_googlenet(no_classes=hid_layer_dim, no_features=no_of_features)
#JAMES_F_0.001_epoch100_0.h5
#deep_rop_bal_cross_lr0.0001_epochs100_k_0.h5 
    # CREATE F&G

    concat_abs_net.load_weights('../Result/DeepROP/deep_rop_bal_cross_lr0.0001' + '_epochs100'+'_k_'+str(kthFold)+ '.h5')

    # Train all models with corresponding images
    concat_abs_net.compile(loss=loss, optimizer=sgd)
    score_3D = concat_abs_net.predict(k_img_test)
    # score = 3.*score_3D[:,0]+2.*score_3D[:,1]+1.*score_3D[:,2]
    scorePlus = score_3D[:,0]
    scoreNormal = score_3D[:,2]
    scorePreplus = score_3D[:,1]
    aucPlus = roc_auc_score(k_rsd_plus,scorePlus)
    aucPreP = roc_auc_score(k_rsd_prep,scorePreplus)
    aucNorm = roc_auc_score(k_rsd_prep,-scoreNormal)
    print('Plus',aucPlus)
    print('Preplus',aucPreP)
    print('Normal',aucNorm)
