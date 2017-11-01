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
from keras import optimizers

'''Trains 3 different G  with absolute labels'''
def biLabels(labels):
    """
    This function will binarized labels.
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
epochs = 100
batch_size = 32  # 1 for validation, 100 for prediction
input_shape = (3, 224, 224)
loss = 'categorical_crossentropy'
sgd = optimizers.SGD(lr=10e-6)

# LOAD DATA FOR ABSOLUTE LABELS
partition_file = pickle.load(open('./6000Partitions.p', 'rb'))
img_folder = 'preprocessed_Peng/'
part_rsd_train = partition_file['RSDTrainPartition']
part_rsd_test = partition_file['RSDTestPartition']
label_rsd = partition_file['RSDLabels']
img_names = partition_file['imgNames']
# kthFold = int(0)
kthFold = int(sys.argv[1])
for k in [kthFold]:
    k_ind_rsd_train = part_rsd_train[k].astype(np.int)
    k_ind_rsd_test = part_rsd_test[k].astype(np.int)
    k_img_train_list = [img_folder+img_names[int(order)]+'.png' for order in k_ind_rsd_train]
    k_img_test_list = [img_folder+img_names[int(order)]+'.png' for order in k_ind_rsd_test]
# Load Images
    # Image for training
    k_img_train_ori = img_to_array(load_img(k_img_train_list[0]))[np.newaxis,:,:,:]
    for img_name_iter in k_img_train_list[1:]:
        img_iter = img_to_array(load_img(img_name_iter))[np.newaxis,:,:,:]
        k_img_train_ori = np.concatenate((k_img_train_ori,img_iter),axis=0)
    k_label_abs_train = label_rsd[k_ind_rsd_train]
    k_label_train_ori = np.reshape(k_label_abs_train,(-1,),order='F')
    k_label_train_bi = biLabels(np.array(k_label_train_ori))
    # Rotate image at each 90 degree and flip images. A single image will show 8 times via its rotations and filps in training.
    k_img_train_rotate_90 = rotate(k_img_train_ori, 90, axes=(2, 3))
    k_img_train_rotate_90_ori = np.concatenate((k_img_train_ori, k_img_train_rotate_90), axis=0)
    k_img_train_rotate_180 = rotate(k_img_train_rotate_90_ori, 180, axes=(2, 3))
    k_img_train_rotate = np.concatenate((k_img_train_rotate_90_ori, k_img_train_rotate_180), axis=0)
    k_img_train_flip = np.flipud(k_img_train_rotate)
    k_img_train_all = np.concatenate((k_img_train_rotate, np.flipud(k_img_train_rotate)), axis=0)
    k_label_train_all = np.tile(k_label_train_bi, (8, 1))
    # Balance the data. Each class uses the same number of images to train.
    k_ind_normal = np.where(k_label_train_all[:,2]==1)[0]
    k_ind_prep = np.where(k_label_train_all[:, 1] == 1)[0]
    k_ind_plus = np.where(k_label_train_all[:, 0] == 1)[0]
    min_size_class = min([len(k_ind_normal),len(k_ind_prep),len(k_ind_plus)])
    np.random.seed(1)
    k_ind_normal_train = k_ind_normal[np.random.choice(len(k_ind_normal),min_size_class)]
    k_ind_prep_train = k_ind_prep[np.random.choice(len(k_ind_prep),min_size_class)]
    k_ind_plus_train = k_ind_plus[np.random.choice(len(k_ind_plus),min_size_class)]
    k_ind_train = np.concatenate((k_ind_normal_train,k_ind_prep_train,k_ind_plus_train))
    k_img_train = k_img_train_all[k_ind_train,:,:,:]
    k_label_train = k_label_train_all[k_ind_train,:]

    # LOAD JAMES' NETWORK FOR F along with ImageNet weights
    concat_abs_net = create_googlenet(no_classes=hid_layer_dim, no_features=no_of_features)
    concat_abs_net.load_weights("googlenet_weights.h5", by_name=True)

    # Train all models with corresponding images
    concat_abs_net.compile(loss=loss, optimizer=sgd)
    concat_abs_net.fit(k_img_train, k_label_train, batch_size=batch_size, epochs=epochs)


    # Save weights for F
    concat_abs_net.save("JAMES_F_10e6_epoch100_32_F_inputAll_" + str(kthFold) + ".h5")
    print("Saved model to disk")