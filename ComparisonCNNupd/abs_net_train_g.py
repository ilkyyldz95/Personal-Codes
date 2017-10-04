from __future__ import absolute_import
from __future__ import print_function
from keras.models import Sequential, Model, model_from_json
from ComparisonCNNupd.createSigLayer import *
from ComparisonCNNupd.createGausLayer import *
from ComparisonCNNupd.createInvSigLayer import *

'''Trains 3 different G  with absolute labels'''
'''No softmax for training'''
def create_network(F, hid_layer_dim):
    # F&G concatenated
    # 3 parallel networks, with the same F and different Gs
    net_plus = Sequential()
    net_plus.add(F)
    net_plus.add(SigLayer(hid_layer_dim))

    net_pre_plus = Sequential()
    net_pre_plus.add(F)
    net_pre_plus.add(GausLayer(hid_layer_dim))

    net_normal = Sequential()
    net_normal.add(F)
    net_normal.add(InvSigLayer(hid_layer_dim))
    return [net_plus,net_pre_plus,net_normal]

# INITIALIZE PARAMETERS
hid_layer_dim = 10 # 1 for James' code: score
epochs = 10
batch_size = 1

# LOAD TRAIN DATA FOR ABSOLUTE LABELS: x_plus, x_pre, x_normal

# LOAD JAMES' NETWORK FOR F, call abs_net
json_file = open('abs_label_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
abs_net = model_from_json(loaded_model_json)
abs_net.load_weights("abs_label_model.h5")
print("F loaded")

# CREATE F&G
[net_plus,net_pre_plus,net_normal] = create_network(abs_net, hid_layer_dim)

# LOAD WEIGHTS OF F INTO THE NEW MODEL
for i in range(len(abs_net.layers)):
    net_plus.layers[0].layers[i].set_weights(abs_net.layers[i].get_weights())
    net_pre_plus.layers[0].layers[i].set_weights(abs_net.layers[i].get_weights())
    net_normal.layers[0].layers[i].set_weights(abs_net.layers[i].get_weights())

# Train all models with corresponding images
net_plus.compile(loss='binary_crossentropy', optimizer='adam')
net_plus.fit(x_plus, np.ones((x_plus.shape[0],1)), epochs = epochs, batch_size=batch_size)

net_pre_plus.compile(loss='binary_crossentropy', optimizer='adam')
net_pre_plus.fit(x_pre, np.ones((x_pre.shape[0],1)), epochs = epochs, batch_size=batch_size)

net_normal.compile(loss='binary_crossentropy', optimizer='adam')
net_normal.fit(x_normal, np.ones((x_normal.shape[0],1)), epochs = epochs, batch_size=batch_size)