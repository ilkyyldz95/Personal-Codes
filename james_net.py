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
from scipy.ndimage import rotate
import matplotlib.pyplot as plt

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

'''def create_network(F, hid_layer_dim, input_shape):
    #F&G concatenated
    #3 parallel networks, with the same F and different Gs

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
    return -K.log(diff)'''

# INITIALIZE PARAMETERS
hid_layer_dim = 3 #F has 1 output: score
no_of_features = 1024
epochs = 10
batch_size = 32     #1 for validation, 100 for prediction
input_shape = (3,224,224)
loss = 'categorical_crossentropy'
optimizer = 'rmsprop'

# LOAD DATA FOR ABSOLUTE LABELS
partition_file = pickle.load(open('./Partitions.p', 'rb'))
img_folder = './preprocessed/All/'
part_rsd_train = partition_file['RSDTrainPlusPartition']
part_rsd_test = partition_file['RSDTestPlusPartition']
label_absolute = partition_file['label13']
label_absolute[label_absolute==1.5] = 2
order_name = partition_file['orderName']
# kthFold = int(0)
kthFold = int(sys.argv[1])
for k in [kthFold]:
    k_ind_rsd_train = part_rsd_train[k]
    k_ind_rsd_test = part_rsd_test[k]
    k_img_train_list = [img_folder+order_name[order+1]+'.png' for order in k_ind_rsd_train]
    k_img_test_list = [img_folder+order_name[order+1]+'.png' for order in k_ind_rsd_test]
# Load Images
    # Image for training
    k_img_train = img_to_array(load_img(k_img_train_list[0]))[np.newaxis,:,:,:]
    for img_name_iter in k_img_train_list[1:]:
        img_iter = img_to_array(load_img(img_name_iter))[np.newaxis,:,:,:]
        k_img_train = np.concatenate((k_img_train,img_iter),axis=0)
    k_img_train_ori = np.tile(k_img_train,[13,1,1,1])
    k_label_abs_train = label_absolute[k_ind_rsd_train,:]
    k_label_train_ori = np.reshape(k_label_abs_train,(-1,),order='F')
    k_label_train_bi = biLabels(np.array(k_label_train_ori))

    # Rotate image at each 90 degree and flip images. A single image will show 8 times via its rotations and filps in training.
    k_img_train_rotate_90 = rotate(k_img_train_ori,90,axes=(2,3))
    k_img_train_rotate_90_ori = np.concatenate((k_img_train_ori,k_img_train_rotate_90),axis=0)
    k_img_train_rotate_180 =  rotate(k_img_train_rotate_90_ori,180,axes=(2,3))
    k_img_train_rotate = np.concatenate((k_img_train_rotate_90_ori,k_img_train_rotate_180),axis=0)
    k_img_train_flip = np.flipud(k_img_train_rotate)
    k_img_train = np.concatenate((k_img_train_rotate,np.flipud(k_img_train_rotate)),axis=0)
    k_label_train = np.tile(k_label_train_bi,(8,1))

# plt.imshow(a.transpose(1,2,0))

# LOAD JAMES' NETWORK FOR F along with ImageNet weights
    F_prev = create_googlenet(no_classes=1000, no_features=no_of_features)
    F_prev.load_weights("googlenet_weights.h5", by_name=True)
    F = create_googlenet(no_classes=hid_layer_dim, no_features=no_of_features)
    for i in range(len(F.layers) - 2):  # last 2 layers depends on the number of classes
        F.layers[i].set_weights(F_prev.layers[i].get_weights())

    # CREATE F&G
    concat_abs_net = F

    # Train all models with corresponding images
    concat_abs_net.compile(loss=loss, optimizer=optimizer)
    concat_abs_net.fit(k_img_train, k_label_train, batch_size=batch_size, epochs=epochs)

    # Save weights for F
    # concat_abs_net.layers[1].save_weights("abs_label_F_"+str(kthFold)+".h5")
    concat_abs_net.save("JAMES_F_save_model_32_F_inputAll_" + str(kthFold) + ".h5")
    print("Saved model to disk")

'''custom_layers = {'PoolHelper': PoolHelper, 'LRN': LRN, 'SigLayer': SigLayer, 'GausLayer': GausLayer,
                 'InvSigLayer': InvSigLayer}
score_function = load_model('abs_label_F.h5', custom_objects=custom_layers)'''
