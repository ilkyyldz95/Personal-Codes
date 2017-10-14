from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

'''G for normal'''
class InvSigLayer(Layer):
    def __init__(self, input_dim, output_dim=1, **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        super(InvSigLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # initiate the tensor variables
        self.a = self.add_weight(name='slope', shape=(input_shape[1], self.output_dim), initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='bias', shape=(1, self.output_dim), initializer='uniform', trainable=True)
        # Create a trainable weight variable for this layer.
        self.trainable_weights = [self.a, self.b]
        super(InvSigLayer, self).build(input_shape)  # built=true

    def call(self, x):
        # 1 - 1/(1 + exp(-ax + b))
        # Dot: matrix multiplication
        return 1 - K.pow(1+K.exp(-K.dot(x, self.a) + np.repeat(self.b, x.shape[0], axis=0)), -1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)