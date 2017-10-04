from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout

'''' LOAD COMPARISON TRAINING AND THEN TRAIN FOR ABSOLUTE LABELS'''

def createAbsLabels(y):
    # y_train has digits from 0 to 9. output for 3 should be [0 0 0 1 0 0 0 0 0 0]
    no_of_classes = 10
    y_vects = [np.zeros((label,)).tolist() + [1.0] + np.zeros((no_of_classes - label - 1,)).tolist() \
                     for label in y]
    return y_vects

def create_network(seq):
    '''f&g concatenated
    '''
    # previous network: f
    model = Sequential()
    for i in range(len(seq.layers)):
        model.add(seq.layers[i])
    # create the follow-up network: g, SAMPLE NETWORK
    model.add(Dense(128, activation='tanh', name = 'new_dense_1'))
    model.add(Dense(128, activation='tanh', name = 'new_dense_2'))
    model.add(Dense(no_of_classes, activation='softmax', name = 'new_dense_3'))
    return model

def compute_abs_accuracy(predictions, labels):
    '''Compute classification accuracy wrt absolute labels
    Predictions are vectors with values between 0 and 1
    '''
    trues = 0
    for i in range(len(labels)): # for each data sample
        currentSample = predictions[i][:].tolist()
        trues = trues + ( currentSample.index(max(currentSample)) == labels[i])
    return 1.*trues / len(labels)

# the data, shuffled and split between train and test sets
# CHANGE FOR OUR CASE: Let smaller numbers correspond to healthier people
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
y_train_vects = createAbsLabels(y_train)
y_test_vects = createAbsLabels(y_test)
epochs = 20
no_of_classes = 10

# load json and corresponding model
json_file = open('comp_label_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
comp_model = model_from_json(loaded_model_json)
comp_model.load_weights("comp_label_model.h5")
seq = comp_model.layers[2] # where trained layers are stored
print("Model loaded")

# create new model using the network for initial layers
total_model = create_network(seq)

# load weights into new model
for i in range(len(seq.layers)):
    total_model.layers[i].set_weights(seq.layers[i].get_weights())

# Train the model, labels vectorized properly
total_model.compile(loss='mean_squared_error', optimizer='adam')
total_model.fit(x_train, y_train_vects, epochs = epochs, batch_size=128)

# Training accuracy
pred = total_model.predict(x_train)
tr_acc = compute_abs_accuracy(pred, y_train)
print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))

'''# serialize model to JSON
model_json = total_model.to_json()
with open("comp_abs_label_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
total_model.save_weights("comp_abs_label_model.h5")
print("Saved model to disk")'''

