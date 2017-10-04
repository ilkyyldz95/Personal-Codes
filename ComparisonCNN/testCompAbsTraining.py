from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import random
from keras.models import model_from_json
from keras import backend as K

''' TEST THE NETWORK TRAINED FOR COMPARISONS FIRST AND ABSOLUTE LABELS LATER'''
def create_test_data(x, y, digit_indices):
    '''Larger digits correspond to higher scores in our model
    '''
    pairs = []
    labels = []
    x_abs = []
    y_abs = []
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
            x_abs += [x[z1]]    # same number of absolute label test data as the number of comparison label data
            y_abs += [y[z1]]

    return np.array(pairs), np.array(labels), np.array(x_abs), np.array(y_abs)

def compute_abs_accuracy(predictions, labels):
    '''Compute classification accuracy wrt absolute labels
    Predictions are vectors with values between 0 and 1
    '''
    trues = 0
    for i in range(len(labels)): # for each data sample
        currentSample = predictions[i][:].tolist()
        trues = trues + ( currentSample.index(max(currentSample)) == labels[i])
    return 1.*trues / len(labels)

def compute_comp_accuracy(predictions, labels):
    '''Compute classification accuracy wrt comparison labels
    Predictions are scalars with values between -1 and 1
    '''
    preds = predictions.ravel() >= 0
    temp = labels[:]
    temp[temp == -1] = 0
    return ((preds & labels).sum() +
            (np.logical_not(preds) & np.logical_not(labels)).sum()) / float(labels.size)

# load data
x_test = np.load('x_test.dat')
y_test = np.load('y_test.dat')
x_test = x_test.reshape(10000, 784)
x_test = x_test.astype('float32')
x_test /= 255
no_of_classes = 10
no_of_f_layers = 4

# create test data
digit_indices = [np.where(y_test == i)[0] for i in range(no_of_classes)]
te_pairs, te_comp, te_x, te_abs = create_test_data(x_test, y_test, digit_indices)

# load json and corresponding model
json_file = open('comp_abs_label_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
total_model = model_from_json(loaded_model_json)
total_model.load_weights("comp_abs_label_model.h5")
print("Model loaded")

# ABS ACCURACY AFTER COMP&ABS TRAINING
pred = total_model.predict(te_x)
te_acc_abs = compute_abs_accuracy(pred, te_abs)
print('* Absolute label accuracy after both trainings: %0.2f%%' % (100 * te_acc_abs))

# COMP ACCURACY AFTER COMP&ABS TRAINING
# get the output of f
f_out = K.function([total_model.layers[0].input],[total_model.layers[no_of_f_layers-1].output])
pred_i = f_out([te_pairs[:, 0]])
pred_j = f_out([te_pairs[:, 1]])
te_acc_comp = compute_comp_accuracy(np.asarray(pred_i)-np.asarray(pred_j), te_comp)
print('* Comparison label accuracy after both trainings: %0.2f%%' % (100 * te_acc_comp))
