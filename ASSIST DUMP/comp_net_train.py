from __future__ import absolute_import
from __future__ import print_function
from keras.layers import Input, Lambda
from keras import optimizers
from keras import backend as K
from googlenet import *
import pickle
import sys
from keras.preprocessing.image import load_img, img_to_array
from scipy.ndimage import rotate
import numpy as np

'''Imports F and trains with comparison labels'''

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
epochs = 10
batch_size = 32 #1 for validation, 100 for prediction
loss = BTLoss
sgd = optimizers.SGD(lr=1e-4)

# LOAD TRAIN DATA FOR COMPARISON LABELS
partition_file = pickle.load(open('./Partitions.p', 'rb'))
'''with open('./Partitions.p', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    partition_file = u.load()'''
img_folder = './preprocessed/All/'
part_cmp_train = partition_file['cmpTrainPlusPartition'] #(fold,pair index)
label_cmp = partition_file['cmpData'] #(expert,imagename1,imagename2,label)
k_ind_image_all = label_cmp[0]  #(imagename1,imagename2, label)

#kthFold = int(0)
kthFold = int(sys.argv[1])
for k in [kthFold]:
    # get all images and labels in kth fold
    k_ind_cmp_train = part_cmp_train[k] #indices corresponding to label_all
    # tuples of compared image names
    k_img_train_list = [ (img_folder+k_ind_image_all[index][0]+'.png', img_folder+k_ind_image_all[index][1]+'.png') for index in k_ind_cmp_train]

    # Load Images
    # k_img_train_1: 1st elm of all pairs,channels,imagex,imagey
    k_img_train_1 = img_to_array(load_img(k_img_train_list[0][0])).astype(np.uint8)[np.newaxis,:,:,:]
    k_img_train_2 = img_to_array(load_img(k_img_train_list[0][1])).astype(np.uint8)[np.newaxis,:,:,:]
    for img_names_iter in k_img_train_list[1:]:
        img_iter_1 = img_to_array(load_img(img_names_iter[0])).astype(np.uint8)[np.newaxis, :, :, :]
        k_img_train_1 = np.concatenate((k_img_train_1, img_iter_1), axis=0)
        img_iter_2 = img_to_array(load_img(img_names_iter[1])).astype(np.uint8)[np.newaxis, :, :, :]
        k_img_train_2 = np.concatenate((k_img_train_2, img_iter_2), axis=0)

    # Replicate for all experts
    k_img_train_ori_1 = np.tile(k_img_train_1,[5,1,1,1])
    k_img_train_ori_2 = np.tile(k_img_train_2, [5,1,1,1])
    k_label_comp_train = [label_cmp[l][index][2] for index in k_ind_cmp_train for l in range(5)]

    # Rotate image at each 90 degree and flip images. A single image will show 8 times via its rotations and filps in training.
    k_img_train_1_rotate_90 = rotate(k_img_train_ori_1,90,axes=(2,3)).astype(np.uint8)
    k_img_train_1_rotate_90_ori = np.concatenate((k_img_train_ori_1,k_img_train_1_rotate_90),axis=0)
    k_img_train_1_rotate_180 = rotate(k_img_train_1_rotate_90_ori,180,axes=(2,3)).astype(np.uint8)
    k_img_train_1_rotate = np.concatenate((k_img_train_1_rotate_90_ori,k_img_train_1_rotate_180),axis=0)
    k_img_train_1_flip = np.flipud(k_img_train_1_rotate).astype(np.uint8)
    k_img_train_1 = np.concatenate((k_img_train_1_rotate,k_img_train_1_flip),axis=0)

    k_img_train_2_rotate_90 = rotate(k_img_train_ori_2, 90, axes=(2, 3)).astype(np.uint8)
    k_img_train_2_rotate_90_ori = np.concatenate((k_img_train_ori_2, k_img_train_2_rotate_90), axis=0)
    k_img_train_2_rotate_180 = rotate(k_img_train_2_rotate_90_ori, 180, axes=(2, 3)).astype(np.uint8)
    k_img_train_2_rotate = np.concatenate((k_img_train_2_rotate_90_ori, k_img_train_2_rotate_180), axis=0)
    k_img_train_2_flip = np.flipud(k_img_train_2_rotate).astype(np.uint8)
    k_img_train_2 = np.concatenate((k_img_train_2_rotate, k_img_train_2_flip), axis=0)

    k_label_train = np.transpose(np.tile(k_label_comp_train,(1,8)))

# LOAD JAMES' NETWORK FOR F
F_prev = create_googlenet(no_classes=1000, no_features=no_of_features)
F_prev.load_weights("googlenet_weights.h5", by_name=True)
F1 = create_googlenet(no_classes=hid_layer_dim, no_features=no_of_features)
F2 = create_googlenet(no_classes=hid_layer_dim, no_features=no_of_features)
for i in range(len(F1.layers) - 2): #last 2 layers depends on the number of classes
    F1.layers[i].set_weights(F_prev.layers[i].get_weights())
    F2.layers[i].set_weights(F_prev.layers[i].get_weights())
print("F loaded")

# CREATE TWIN NETWORKS: Siamese
# because we re-use the same instance `base_network`, the weights of the network will be shared across the two branches
input_a = F1.input
input_b = F2.input

processed_a = F1(input_a)
processed_b = F2(input_b)

distance = Lambda(BTPred, output_shape=(1,))([processed_a, processed_b])

comp_net = Model([input_a, input_b], distance)

# train
comp_net.compile(loss=BTLoss, optimizer=sgd)
comp_net.fit([k_img_train_1, k_img_train_2], k_label_train, batch_size=batch_size, epochs=epochs)

# Save weights for F
comp_net.save("comp_label_F_32_F_inputAll_" + str(kthFold) + ".h5")
print("Saved model to disk")


