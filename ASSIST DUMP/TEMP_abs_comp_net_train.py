from __future__ import absolute_import
from __future__ import print_function
import sys
from keras.layers import Input, Activation, merge, Lambda
from keras.models import Model, load_model
from createSigLayer import SigLayer
from createGausLayer import GausLayer
from createInvSigLayer import InvSigLayer
import numpy as np
from googlenet import create_googlenet
from keras import backend as K
import pickle
from keras.preprocessing.image import load_img, img_to_array
from scipy.ndimage import rotate
from keras import optimizers

'''Trains whole network with absolute and comparison labels simultaneously'''
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

def create_abs_network(F, hid_layer_dim):
    ''' F&G concatenated
    3 parallel networks, with the same F and different Gs'''

    shared_input = F.input
    # Pass input through F
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
    return -K.log(diff)

def BTLoss(y_true, y_pred):
    """
    Negative log likelihood of Bradley-Terry Penalty, to be minimized. y = beta.*x
    """
    exponent = K.exp(-y_true * (y_pred))
    return K.log(1 + exponent)

def totalLoss(y_true, y_pred):
    # y_pred = y_i(,3), y_j(,3), y_i_j(,1)
    # returns alpha*(La(y_i) + La(y_j))+(1-alpha)*Lc(y_i_j)
    return alpha*( absLoss(y_true[:3],y_pred[:3]) + absLoss(y_true[3:6],y_pred[3:6]) ) \
           + (1-alpha)*BTLoss(y_true[6],y_pred[6])

# INITIALIZE PARAMETERS
hid_layer_dim = 1 #score
no_of_features = 1024
epochs = 100
batch_size = 32
input_shape = (3,224,224)
loss = totalLoss
sgd = optimizers.SGD(lr=1e-10)
alpha = 0.5 # balance between absolute and comparison contributions

# LOAD DATA FOR ALL LABELS
partition_file = pickle.load(open('./Partitions.p', 'rb'))
'''with open('./Partitions.p', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    partition_file = u.load()'''
img_folder = './preprocessed/All/'

# kthFold = int(0)
kthFold = int(sys.argv[1])
# Load absolute labels
part_rsd_train = partition_file['RSDTrainPlusPartition']
label_absolute = partition_file['label13']
label_absolute[label_absolute==1.5] = 2
order_name = partition_file['orderName']
for k in [kthFold]:
    k_ind_rsd_train = part_rsd_train[k]
    k_img_train_list = [img_folder+order_name[order+1]+'.png' for order in k_ind_rsd_train]
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
    k_img_train_abs = np.concatenate((k_img_train_rotate,np.flipud(k_img_train_rotate)),axis=0)
    k_label_train_abs = np.tile(k_label_train_bi,(8,1))

# Load comparison labels
part_cmp_train = partition_file['cmpTrainPlusPartition']  # (fold,pair index)
label_cmp = partition_file['cmpData']  # (expert,imagename1,imagename2,label)
k_ind_image_all = label_cmp[0]  # (imagename1,imagename2, label)
for k in [kthFold]:
    # get all images and labels in kth fold
    k_ind_cmp_train = part_cmp_train[k]  # indices corresponding to label_all
    # tuples of compared image names
    k_img_train_list = [
        (img_folder + k_ind_image_all[index][0] + '.png', img_folder + k_ind_image_all[index][1] + '.png') for index
        in k_ind_cmp_train]
    # Load Images
    # k_img_train_1: 1st elm of all pairs,channels,imagex,imagey
    k_img_train_1 = img_to_array(load_img(k_img_train_list[0][0])).astype(np.uint8)[np.newaxis, :, :, :]
    k_img_train_2 = img_to_array(load_img(k_img_train_list[0][1])).astype(np.uint8)[np.newaxis, :, :, :]
    for img_names_iter in k_img_train_list[1:]:
        img_iter_1 = img_to_array(load_img(img_names_iter[0])).astype(np.uint8)[np.newaxis, :, :, :]
        k_img_train_1 = np.concatenate((k_img_train_1, img_iter_1), axis=0)
        img_iter_2 = img_to_array(load_img(img_names_iter[1])).astype(np.uint8)[np.newaxis, :, :, :]
        k_img_train_2 = np.concatenate((k_img_train_2, img_iter_2), axis=0)
    # Replicate for all experts
    k_img_train_ori_1 = np.tile(k_img_train_1, [5, 1, 1, 1])
    k_img_train_ori_2 = np.tile(k_img_train_2, [5, 1, 1, 1])
    k_label_comp_train = [label_cmp[l][index][2] for index in k_ind_cmp_train for l in range(5)]
    # Rotate image at each 90 degree and flip images. A single image will show 8 times via its rotations and filps in training.
    k_img_train_1_rotate_90 = rotate(k_img_train_ori_1, 90, axes=(2, 3)).astype(np.uint8)
    k_img_train_1_rotate_90_ori = np.concatenate((k_img_train_ori_1, k_img_train_1_rotate_90), axis=0)
    k_img_train_1_rotate_180 = rotate(k_img_train_1_rotate_90_ori, 180, axes=(2, 3)).astype(np.uint8)
    k_img_train_1_rotate = np.concatenate((k_img_train_1_rotate_90_ori, k_img_train_1_rotate_180), axis=0)
    k_img_train_1_flip = np.flipud(k_img_train_1_rotate).astype(np.uint8)
    k_img_train_comp_1 = np.concatenate((k_img_train_1_rotate, k_img_train_1_flip), axis=0)

    k_img_train_2_rotate_90 = rotate(k_img_train_ori_2, 90, axes=(2, 3)).astype(np.uint8)
    k_img_train_2_rotate_90_ori = np.concatenate((k_img_train_ori_2, k_img_train_2_rotate_90), axis=0)
    k_img_train_2_rotate_180 = rotate(k_img_train_2_rotate_90_ori, 180, axes=(2, 3)).astype(np.uint8)
    k_img_train_2_rotate = np.concatenate((k_img_train_2_rotate_90_ori, k_img_train_2_rotate_180), axis=0)
    k_img_train_2_flip = np.flipud(k_img_train_2_rotate).astype(np.uint8)
    k_img_train_comp_2 = np.concatenate((k_img_train_2_rotate, k_img_train_2_flip), axis=0)

    k_label_train_comp = np.transpose(np.tile(k_label_comp_train, (1, 8)))

# LOAD JAMES' NETWORK FOR F and INITIALIZE WITH IMAGENET WEIGHTS
F_prev = create_googlenet(no_classes=1000, no_features=no_of_features)
F_prev.load_weights("googlenet_weights.h5", by_name=True)
F1 = create_googlenet(no_classes=hid_layer_dim, no_features=no_of_features)
F2 = create_googlenet(no_classes=hid_layer_dim, no_features=no_of_features)
for i in range(len(F1.layers) - 2): #last 2 layers depends on the number of classes
    F1.layers[i].set_weights(F_prev.layers[i].get_weights())
    F2.layers[i].set_weights(F_prev.layers[i].get_weights())
print("F loaded")

# Rename F2 layers
for layer in F2.layers:
    layer.name = layer.name + '_rep'

# CREATE F&G
abs_net_1 = create_abs_network(F1, hid_layer_dim) #output shape:3, 1 for each custom layer
abs_net_2 = create_abs_network(F2, hid_layer_dim)

# CREATE SIAMESE NETWORK
score_i = abs_net_1.get_layer('prob_modified').output
score_j = abs_net_2.get_layer('prob_modified_rep').output

y_i_j = Lambda(BTPred, output_shape=(1,))([score_i, score_j])
y_i = abs_net_1.output
y_j = abs_net_2.output

# CONCATENATE MODELS
input_i = abs_net_1.input
input_j = abs_net_2.input
abs_comp_net = Model([input_i, input_j], [y_i, y_j, y_i_j])

# Train all models with corresponding images
abs_comp_net.compile(loss=loss, optimizer=sgd)
concat_abs_net.fit(k_img_train_abs, k_label_train_abs,
                   batch_size=batch_size, epochs=epochs)

# Save weights for F
# concat_abs_net.layers[1].save_weights("abs_label_F_"+str(kthFold)+".h5")
concat_abs_net.save("abs_comp_label_" + str(kthFold) + ".h5")
print("Saved model to disk")