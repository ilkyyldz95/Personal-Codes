from keras.utils.np_utils import to_categorical
from keras.utils.common import *
from keras.utils.plotting import *
from os.path import join
import numpy as np
from keras.utils.common import *
from ..utils.image import create_generator
from ComparisonCNNupd.googlenet import create_googlenet

class RetiNet(object):
    def __init__(self, train_data, val_data, loss, optimizer, batch_size = 32, no_of_classes = 1, no_of_features = 1024, epochs = 50):
        self.train_data = train_data
        self.val_data = val_data
        self.no_of_classes = no_of_classes
        self.no_of_features = no_of_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer = optimizer

    def train(self):
        # only for google net, compile
        self.model = create_googlenet(self.no_of_classes, self.no_of_features)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])

        # Train
        input_shape = self.model.input_shape[1:]

        # Create generators
        train_gen, _, _ = create_generator(self.train_data, input_shape, training=True, batch_size=self.batch_size)

        if self.val_data is not None:
            val_gen, _, _ = create_generator(self.val_data, input_shape, training=False, batch_size=1)
            no_val_samples = val_gen.x.shape[0]
        else:
            print
            "No validation data provided."
            val_gen = None
            no_val_samples = None

        history = self.model.fit_generator(
            train_gen,
            samples_per_epoch=train_gen.x.shape[0],
            nb_epoch=self.epochs,
            validation_data=val_gen,
            nb_val_samples=no_val_samples)

        # Save model arch, weights and history
        self.model.save_weights(join(self.experiment_dir, 'final_weights.h5'))

        with open(join(self.experiment_dir, 'model_arch.json'), 'w') as arch:
            arch.writelines(self.model.to_json())

        # serialize weights to HDF5
        self.model.save_weights("F.h5")
        print("Saved model to disk")

    def predict(self, img_arr):

        return self.model.predict(img_arr, batch_size=100)

    def evaluate(self, data_path, n_samples=None):

        datagen, y_true, class_indices = create_generator(data_path, self.model.input_shape[1:],
                                                          batch_size=1, training=False)
        if not n_samples:
            n_samples = datagen.X.shape[0] #X.shape[0]

        predictions = self.model.predict_generator(datagen, n_samples)
        data_dict = {'data': datagen, 'classes': class_indices, 'y_true': to_categorical(y_true[:n_samples]), 'y_pred': predictions}

        cols = np.asarray(sorted([[k, v] for k, v in class_indices.items()], key=lambda x: x[1]))
        # pred_df = pd.DataFrame(data=predictions, columns=cols[:, 0])
        # true_df = pd.DataFrame(data=to_categorical(y_true), columns=cols[:, 0])
        #
        # pred_df.to_csv(join(self.eval_dir, 'predictions.csv'))
        # true_df.to_csv(join(self.eval_dir, 'ground_truth.csv'))

        return data_dict

if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', required=True)
    parser.add_argument('-d', '--data', dest='data', default=None)

    args = parser.parse_args()

    # Instantiate model and train
    r = RetiNet(train_data, val_data, loss, optimizer, batch_size = 32, no_of_classes = 1, no_of_features = 1024, epochs = 50)
    r.train()
    data_dict = r.evaluate(args.data)
        # calculate_metrics(data_dict, out_dir=r.eval_dir)