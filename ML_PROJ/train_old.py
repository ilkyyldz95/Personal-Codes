from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from scipy.ndimage import imread
from sklearn.metrics import roc_auc_score

from os import listdir
import numpy as np
from time import time

# initialization
# CHANNELS LAST!!!
input_shape = (335, 472, 3)
data_path_pos = './positive/'
data_path_neg = './negative/'
num_imag_pos = len(listdir(data_path_pos))
num_imag_neg = len(listdir(data_path_neg))
batch_size = 10
epochs = 25

# create training data
x_train = np.zeros((num_imag_pos + num_imag_neg, 335, 472, 3), dtype=np.uint8)
count = 0
for img in listdir(data_path_pos):
    temp = imread(data_path_pos + img, flatten=False, mode=None).astype(np.uint8)
    if temp.shape != input_shape:
        temp = np.tile(temp[:, :, np.newaxis], (1, 1, 3))
    x_train[count, :, :, :] = temp
    count += 1

for img in listdir(data_path_neg):
    temp = imread(data_path_neg + img, flatten=False, mode=None).astype(np.uint8)
    if temp.shape != input_shape:
        temp = np.tile(temp[:, :, np.newaxis], (1, 1, 3))
    x_train[count, :, :, :] = temp
    count += 1

y_train = np.concatenate((np.ones(num_imag_pos), np.zeros(num_imag_neg)))

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer
predictions = Dense(1, activation='sigmoid')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# train
model.compile(optimizer='sgd', loss='binary_crossentropy')

t_0 = time()
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
t_1 = time()

# test
predictions = model.predict(x_train)
print('AUC:'+str(roc_auc_score(y_train, predictions)))
print('Training time:'+str(t_1-t_0))
model.save('model_bs_'+str(batch_size)+'_ep_'+str(epochs)+'.h5')
