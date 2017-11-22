from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import keras.initializers

'''G for plus disease'''
class SigLayer(Layer):
    def __init__(self, input_dim = 1, output_dim=1, **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        super(SigLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # initiate the tensor variables
        self.a = self.add_weight(name='slope', shape=(input_shape[1], self.output_dim),
                                 initializer=keras.initializers.Constant(value=1),
                                 trainable=True)
        self.b = self.add_weight(name='bias', shape=(1, input_shape[1]),
                                 initializer=keras.initializers.Constant(value=0.5),
                                 trainable=True)
        # Create a trainable weight variable for this layer.
        self.trainable_weights = [self.a, self.b]
        super(SigLayer, self).build(input_shape)  # built=true

    def call(self, x):
        # 1/(1 + exp(a(x - b)))
        # Dot: matrix multiplication & repeat weights for batch size adaptation
        return 1. / (1+K.exp(K.dot(x - self.b, self.a)))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    '''def get_config(self):
        config = {"a": self.a,
                  "b": self.b,
                  "output_dim": self.output_dim,
                  "input_dim": self.input_dim,
                  "trainable_weights": self.trainable_weights}
        base_config = super(SigLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))'''