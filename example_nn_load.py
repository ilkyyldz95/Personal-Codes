from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.txt", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# get the output of the layer before sigmoid, which will be the input of the follow-up network
new_input = loaded_model.predict(X)

# train a follow-up model
model = Sequential()
model.add(Dense(12, input_dim=1, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(new_input, Y, epochs=20, batch_size=1)

# calculate predictions
predictions = model.predict(new_input)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)