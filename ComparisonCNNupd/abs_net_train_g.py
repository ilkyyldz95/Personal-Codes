from __future__ import absolute_import
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Input, Lambda
from ComparisonCNNupd.createSigLayer import *
from ComparisonCNNupd.createGausLayer import *
from ComparisonCNNupd.createInvSigLayer import *
from PIL import Image
import numpy as np
import glob
from ComparisonCNNupd.googlenet import *

'''Trains 3 different G  with absolute labels'''
def mySoftMax(inputs):
    out_norm = K.exp(inputs[0])/K.sum(K.exp(inputs))
    out_pre = K.exp(inputs[1]) / K.sum(K.exp(inputs))
    out_plus = K.exp(inputs[2]) / K.sum(K.exp(inputs))
    return K.stack([out_norm, out_pre, out_plus])

def create_network(F, hid_layer_dim, input_shape):
    ''' F&G concatenated
    3 parallel networks, with the same F and different Gs'''

    # CREATE G
    net_plus = Sequential()
    net_plus.add(F)
    net_plus.add(SigLayer(hid_layer_dim))

    net_pre_plus = Sequential()
    net_pre_plus.add(F)
    net_pre_plus.add(GausLayer(hid_layer_dim))

    net_normal = Sequential()
    net_normal.add(F)
    net_normal.add(InvSigLayer(hid_layer_dim))

    # LOAD INITIAL WEIGHTS OF F INTO THE NEW MODEL
    for i in range(len(F.layers)):
        net_plus.layers[0].layers[i].set_weights(F.layers[i].get_weights())
        net_pre_plus.layers[0].layers[i].set_weights(F.layers[i].get_weights())
        net_normal.layers[0].layers[i].set_weights(F.layers[i].get_weights())

    # Construct the whole model with 3 dimensional softmax output
    processed_norm = net_normal(Input(shape=input_shape))
    processed_pre = net_pre_plus(Input(shape=input_shape))
    processed_plus = net_plus(Input(shape=input_shape))

    concat_abs_net = Model(Input(shape=input_shape), [processed_norm, processed_pre, processed_plus])
    concat_abs_net.add(Activation('softmax', input_shape = (3,)))

    return concat_abs_net

def absLoss(y_true, y_pred):
    """
    Negative log likelihood of absolute model
    y_true = 0,1,2
    y_pred = soft[g_0(s), g_1(s), g_2(s)]
    Take the g output corresponding to the label
    """
    if y_true == 0:
        diff = y_pred[0]
    elif y_true == 1:
        diff = y_pred[1]
    else:
        diff = y_pred[2]
    return -K.log(diff)

# INITIALIZE PARAMETERS
hid_layer_dim = 1 #score
input_shape = (3,224,224)
no_of_images = 196
no_of_test_images = 64
no_of_features = 1024
epochs = 10
batch_size = 32
loss = absLoss
optimizer = 'sgd'

# LOAD TEST DATA FOR ABSOLUTE LABELS
images = np.zeros((no_of_images,3,224,224))
shuff_images = np.zeros((no_of_images,3,224,224))
abs_labels = np.zeros((no_of_images,))
shuff_abs_labels = np.zeros((no_of_images,))
count = 0

for filename in glob.glob('../ComparisonCNNupd/vessels/*.png'):
    im = np.asarray(Image.open(filename).convert('L'))
    images[count][:][:][:] = im[128:352,208:432]
    count += 1
# 0 for normal, 1 for pp, 2 for p
abs_labels[:31] = 1
abs_labels[31:48] = 2
abs_labels[101:135] = 1
abs_labels[135:149] = 2

# shuffle the data
p = np.random.permutation(no_of_images)
for im in range(no_of_images):
    shuff_images[im][:][:][:] = images[p[im]][:][:][:]
    shuff_abs_labels[im] = abs_labels[p[im]]
test_im = shuff_images[:no_of_test_images][:][:][:]
test_label = shuff_abs_labels[:no_of_test_images]

# LOAD JAMES' NETWORK FOR F, call abs_net
abs_net = create_googlenet(no_classes=hid_layer_dim, no_features=no_of_features)

# CREATE F&G
concat_abs_net = create_network(abs_net, hid_layer_dim, input_shape)

# Train all models with corresponding images
concat_abs_net.compile(loss=loss, optimizer=optimizer)
concat_abs_net.fit(test_im, test_label, batch_size=batch_size, epochs=epochs)

# Save weights for F
concat_abs_net.layers[0].save_weights("abs_label_F.h5")
print("Saved model to disk")