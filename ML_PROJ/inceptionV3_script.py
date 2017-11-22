from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from os import listdir
import numpy as np
from time import time
from scipy.ndimage import imread

# initialization
# CHANNEL number is the last index!!!
input_shape = (335, 472, 3)
data_path_pos = 'DATASET/positive/'
data_path_neg = 'DATASET/negative/'
num_imag_pos = len(listdir(data_path_pos))/2
num_imag_neg = len(listdir(data_path_neg))/2
batch_size = 10
epochs = 1
optimizer = 'sgd'

# create training data
x_train = np.zeros((num_imag_pos + num_imag_neg, 335, 472, 3), dtype=np.uint8)
count = 0
for img in range(num_imag_pos):
    temp = imread(data_path_pos + listdir(data_path_pos)[img]).astype(np.uint8)
    if temp.shape != input_shape:
        temp = np.tile(temp[:, :, np.newaxis], (1, 1, 3))
    x_train[count, :, :, :] = temp
    count += 1

for img in range(num_imag_neg):
    temp = imread(data_path_neg + listdir(data_path_neg)[img]).astype(np.uint8)
    if temp.shape != input_shape:
        temp = np.tile(temp[:, :, np.newaxis], (1, 1, 3))
    x_train[count, :, :, :] = temp
    count += 1

y_train = np.concatenate((np.ones(num_imag_pos), np.zeros(num_imag_neg)))

print 'Dataset is ready to be processed. Creating InceptionV3 model...\n'
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

# Compile model
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['binary_accuracy'])

print 'InceptionV3 ready, starting training...\n'
# Train model
t_0 = time()
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
t_1 = time()

# Training error
print 'Binary Accuracy (correct decision % / 100):', model.evaluate(x_train, y_train)[1]
print 'Training time in seconds:'+str(t_1-t_0)
#print 'Saving model in current directory as:', 'model_bs_'+str(batch_size)+'_ep_'+str(epochs)+'.h5'
#model.save('model_bs_'+str(batch_size)+'_ep_'+str(epochs)+'.h5')
