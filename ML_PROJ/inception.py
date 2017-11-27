from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from time import time


def inception(x_train, y_train, x_test, y_test, save_model=None):

    input_shape = (335, 472, 3)
    batch_size = 10
    epochs = 1
    optimizer = 'sgd'

    # create the base pre-trained model
    print 'Creating InceptionV3...\n'
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)

    # add a global spatial average pooling layer
    print 'Adding additional layers...\n'
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)

    # and a logistic layer
    predictions = Dense(1, activation='sigmoid')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile model
    print 'Compiling final model...\n'
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['binary_accuracy'])

    # Train model
    print 'InceptionV3 ready, starting training...\n'
    t_0 = time()
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    training_time = time() - t_0
    print 'Training finished!'

    # Test. Then, display some results
    accuracy_metric = model.evaluate(x_test, y_test)[1]

    print 'Binary Accuracy (correct decision % / 100):', accuracy_metric
    print 'Training time in hours:'+str(training_time/3600.)


    print 'Saving model in current directory as:', '1st_half_fold_index_' + str(save_model) + '.h5'
    model.save('1st_half_fold_index_' + str(save_model) + '.h5')

    return accuracy_metric, training_time