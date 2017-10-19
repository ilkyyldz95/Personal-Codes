from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from constraints import *
import keras.initializers

'''G for pre-plus disease'''
class GausLayer(Layer):
    def __init__(self, input_dim, output_dim=1, **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        super(GausLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # initiate the tensor variables
        self.mu = self.add_weight(name='mean', shape=(1, input_shape[1]),
                                  initializer=keras.initializers.Constant(value=0.5),
                                  trainable=True)
        self.var = self.add_weight(name='variance', shape=(1, self.output_dim),
                                   initializer=keras.initializers.Constant(value=0.1),
                                   trainable=True, constraint=PosVal())
        # Create a trainable weight variable for this layer.
        self.trainable_weights = [self.mu, self.var]
        super(GausLayer, self).build(input_shape)  # built=true

    def call(self, x):
        # Gaussian with mu and var
        # *: elementwise & repeat weights for batch size adaptation
        return K.exp(-((x - self.mu)**2)
                     / (2 * self.var)) \
                    * 1. / (K.sqrt(2 * np.pi * self.var) + K.epsilon())

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)