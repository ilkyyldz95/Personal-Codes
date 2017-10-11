from __future__ import absolute_import
from __future__ import print_function
from keras.layers import Input
from ComparisonCNNupd.createSigLayer import *
from ComparisonCNNupd.createGausLayer import *
from ComparisonCNNupd.createInvSigLayer import *
from PIL import Image
import numpy as np
import glob
from ComparisonCNNupd.googlenet import *
from keras import backend as K
from theano.tensor import nonzero_values

'''Trains 3 different G  with absolute labels'''
def create_network(F, hid_layer_dim, input_shape):
    ''' F&G concatenated
    3 parallel networks, with the same F and different Gs'''

    shared_input = Input(shape=input_shape)
    # LOAD INITIAL WEIGHTS OF F INTO THE NEW MODEL
    for i in range(len(F.layers)):
        F.layers[i].set_weights(F.layers[i].get_weights())

    f_out = F(shared_input)
    # Create G
    processed_norm = InvSigLayer(hid_layer_dim)(f_out)
    processed_pre = GausLayer(hid_layer_dim)(f_out)
    processed_plus = SigLayer(hid_layer_dim)(f_out)

    # Add 3 dimensional softmax output
    activ_norm = Activation('softmax')(processed_norm)
    activ_pre = Activation('softmax')(processed_pre)
    activ_plus = Activation('softmax')(processed_plus)

    # Create the whole network
    concat_abs_net = Model(shared_input, [activ_norm, activ_pre, activ_plus])

    return concat_abs_net

def absLoss(y_true, y_pred):
    """
    Negative log likelihood of absolute model
    y_true = [100],[010],[001]
    y_pred = soft[g_0(s), g_1(s), g_2(s)]
    Take the g output corresponding to the label
    """
    diff = nonzero_values(y_pred * y_true)
    return -K.log(diff)

# INITIALIZE PARAMETERS
hid_layer_dim = 1 #score
input_shape = (3,224,224)
no_of_images = 196
no_of_test_images = 2
no_of_features = 1024
epochs = 10
batch_size = 1
loss = absLoss
optimizer = 'sgd'

# LOAD TEST DATA FOR ABSOLUTE LABELS
images = np.zeros((no_of_images,3,224,224))
shuff_images = np.zeros((no_of_images,3,224,224))
abs_labels = np.zeros((no_of_images,3))
shuff_abs_labels = np.zeros((no_of_images,3))
count = 0
for filename in glob.glob('../ComparisonCNNupd/vessels/*.png'):
    im = np.asarray(Image.open(filename).convert('L'))
    images[count][:][:][:] = im[128:352,208:432]
    count += 1
# 100 for normal, 010 for pp, 001 for p
abs_labels[:31,:] = [0,1,0]
abs_labels[31:48,:] = [0,0,1]
abs_labels[48:101,:] = [1,0,0]
abs_labels[101:135,:] = [0,1,0]
abs_labels[135:149,:] = [0,0,1]
abs_labels[149:,:] = [1,0,0]
# shuffle the data
p = np.random.permutation(no_of_images)
for im in range(no_of_images):
    shuff_images[im][:][:][:] = images[p[im]][:][:][:]
    shuff_abs_labels[im,:] = abs_labels[p[im],:]
test_im = shuff_images[:no_of_test_images][:][:][:]
test_label = shuff_abs_labels[:no_of_test_images,:]

# LOAD JAMES' NETWORK FOR F, call abs_net
abs_net = create_googlenet(no_classes=hid_layer_dim, no_features=no_of_features)

# CREATE F&G
concat_abs_net = create_network(abs_net, hid_layer_dim, input_shape)

# Train all models with corresponding images
concat_abs_net.compile(loss=loss, optimizer=optimizer)
concat_abs_net.fit(test_im, [test_label[:,0], test_label[:,1], test_label[:,2]], batch_size=batch_size, epochs=epochs)

# Save weights for F
concat_abs_net.layers[1].save_weights("abs_label_F.h5")
print("Saved model to disk")