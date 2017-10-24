from __future__ import absolute_import
from __future__ import print_function
from keras.layers import Input, Activation, merge
from keras.models import Model, load_model
from createSigLayer import SigLayer
from createGausLayer import GausLayer
from createInvSigLayer import InvSigLayer
import numpy as np
from googlenet import create_googlenet
from keras import backend as K
import pickle
from keras.preprocessing.image import load_img, img_to_array
from googlenet_custom_layers import PoolHelper, LRN

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

def create_network(F, hid_layer_dim):
    ''' F&G concatenated
    3 parallel networks, with the same F and different Gs'''

    shared_input = F.input
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
input_shape = (3,224,224)
no_of_features = 1024
epochs = 10
batch_size = 32
loss = absLoss
optimizer = 'adam'

# LOAD DATA FOR ABSOLUTE LABELS
#partition_file = pickle.load(open('./Partitions.p', 'rb'))
with open('./Partitions.p', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    partition_file = u.load()
img_folder = './preprocessed/All/'
part_rsd_train = partition_file['RSDTrainPlusPartition']
part_rsd_test = partition_file['RSDTestPlusPartition']
label_absolute = partition_file['label13']
label_absolute[label_absolute==1.5] = 2
order_name = partition_file['orderName']
for k in range(1):
    k_ind_rsd_train = part_rsd_train[k]
    k_ind_rsd_test = part_rsd_test[k]
    k_img_train_list = [img_folder+order_name[order+1]+'.png' for order in k_ind_rsd_train]
    k_img_test_list = [img_folder+order_name[order+1]+'.png' for order in k_ind_rsd_test]
# Load Images
    # Image for training
    k_img_train = img_iter = img_to_array(load_img(k_img_train_list[0]))[np.newaxis,:,:,:]
    for img_name_iter in k_img_train_list[1:]:
        img_iter = img_to_array(load_img(img_name_iter))[np.newaxis,:,:,:]
        k_img_train = np.concatenate((k_img_train,img_iter),axis=0)
    k_img_train = np.tile(k_img_train,[13,1,1,1])
    k_label_abs_train = label_absolute[k_ind_rsd_train,:]
    k_label_train_ori = np.reshape(k_label_abs_train,(-1,),order='F')
    k_label_train = biLabels(np.array(k_label_train_ori))

# LOAD JAMES' NETWORK FOR F along with ImageNet weights
F_prev = create_googlenet(no_classes=1000, no_features=no_of_features)
F_prev.load_weights("googlenet_weights.h5", by_name=True)
F = create_googlenet(no_classes=hid_layer_dim, no_features=no_of_features)
for i in range(len(F.layers) - 2): #last 2 layers depends on the number of classes
        F.layers[i].set_weights(F_prev.layers[i].get_weights())

# CREATE F&G
concat_abs_net = create_network(F, hid_layer_dim)

# Train all models with corresponding images
concat_abs_net.compile(loss=loss, optimizer=optimizer)
concat_abs_net.fit(k_img_train, k_label_train, batch_size=batch_size, epochs=epochs)

# Save weights for F
concat_abs_net.layers[1].save_weights("abs_label_F.h5")
print("Saved model to disk")

'''custom_layers = {'PoolHelper': PoolHelper, 'LRN': LRN, 'SigLayer': SigLayer, 'GausLayer': GausLayer,
                 'InvSigLayer': InvSigLayer}
score_function = load_model('abs_label_F.h5', custom_objects=custom_layers)'''