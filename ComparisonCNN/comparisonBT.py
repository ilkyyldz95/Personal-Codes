from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import random
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import RMSprop
from keras import backend as K

'''' TRAIN FOR COMPARisON LABELS'''

def BTPred(scalars):
    """
    This is the output when we predict comparison labels.
    """
    s1 = scalars[0]
    s2 = scalars[1]
    #    s1, s2 = scalars (beta.*x)
    return s1 - s2


def BTLoss(y_true, y_pred):
    """
    Negative log likelihood of Bradley-Terry Penalty, to be minimized
    y = beta.*x
    """
    exponent = K.exp(-y_true * (y_pred))
    return K.log(1 + exponent)

def create_pairs(x, digit_indices):
    '''Larger digits correspond to higher scores in our model
    '''
    pairs = []
    labels = []
    no_of_classes = 10
    n = min([len(digit_indices[d]) for d in range(no_of_classes)]) - 1
    for d in range(no_of_classes):
        for i in range(n):
            inc = random.randrange(0, no_of_classes)
            dn = (d + inc) % no_of_classes
            if d == dn: #si == sj
                z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
                pairs += [[x[z1], x[z2]]]
                labels += [1]
            elif d > dn: #si > sj
                z1, z2 = digit_indices[d][i], digit_indices[dn][i]
                pairs += [[x[z1], x[z2]]]
                labels += [1]
            else:
                z1, z2 = digit_indices[d][i], digit_indices[dn][i]
                pairs += [[x[z1], x[z2]]]
                labels += [-1]

    return np.array(pairs), np.array(labels)

def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    ACTIVATION CHANGED TO TANH
    '''
    seq = Sequential()
    seq.add(Dense(128, input_shape=(input_dim,), activation='tanh'))
    #seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='tanh'))
    #seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='tanh'))
    seq.add(Dense(1, activation='tanh'))
    return seq

def compute_comp_accuracy(predictions, labels):
    '''Compute classification accuracy wrt comparison labels
    Predictions are scalars with values between -1 and 1
    '''
    preds = predictions.ravel() >= 0
    temp = labels[:]
    temp[temp == -1] = 0
    return ((preds & labels).sum() +
            (np.logical_not(preds) & np.logical_not(labels)).sum()) / float(labels.size)

# the data, shuffled and split between train and test sets
# CHANGE FOR OUR CASE: Let smaller numbers correspond to healthier people
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.load('x_train.dat')
y_train = np.load('y_train.dat')
x_test = np.load('x_test.dat')
y_test = np.load('y_test.dat')
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
input_dim = 784
epochs = 10
no_of_classes = 10

# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(no_of_classes)]
tr_pairs, tr_y = create_pairs(x_train, digit_indices)

digit_indices = [np.where(y_test == i)[0] for i in range(no_of_classes)]
te_pairs, te_y = create_pairs(x_test, digit_indices)

# network definition
base_network = create_base_network(input_dim)

input_a = Input(shape=(input_dim,))
input_b = Input(shape=(input_dim,))

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(BTPred,
                  output_shape=(1,))([processed_a, processed_b])

model = Model([input_a, input_b], distance)

# train
rms = RMSprop()
model.compile(loss=BTLoss, optimizer=rms)
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=128,
          epochs=epochs,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

# Training accuracy
pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_comp_accuracy(pred, tr_y)
print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))


'''# serialize model to JSON
model_json = model.to_json()
with open("comp_label_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("comp_label_model.h5")
print("Saved model to disk")'''