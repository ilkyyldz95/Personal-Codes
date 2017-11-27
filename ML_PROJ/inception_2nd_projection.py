from keras.models import load_model
from time import time


def inception_2nd_projection(x_train, y_train, x_test, y_test, save_model=None):
    batch_size = 10
    epochs = 1

    # Load model trained for 1st half
    print 'Loading 1st half model from directory:', '1st_half_fold_index_' + str(save_model) + '.h5'
    model = load_model('1st_half_fold_index_' + str(save_model) + '.h5')

    # Train model
    print 'InceptionV3 ready, starting training...\n'
    t_0 = time()
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    training_time = time() - t_0
    print 'Training finished!'

    # Test. Then, display some results
    accuracy_metric = model.evaluate(x_test, y_test)[1]

    print 'Binary Accuracy (correct decision % / 100):', accuracy_metric
    print 'Training time in hours:' + str(training_time / 3600.)

    print 'Saving model in current directory as:', '2nd_half_fold_index_' + str(save_model) + '.h5'
    model.save('2nd_half_fold_index_' + str(save_model) + '.h5')
