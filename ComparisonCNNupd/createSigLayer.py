from keras import backend as K
from keras.engine.topology import Layer

'''G for plus disease'''
class SigLayer(Layer):
    def __init__(self, input_dim, output_dim=1, **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        super(SigLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # initiate the tensor variables
        self.a = self.add_weight(name = 'slope', shape=(input_shape[1], self.output_dim), initializer='uniform', trainable=True)
        self.b = self.add_weight(name = 'bias', shape=(1, self.output_dim), initializer='uniform', trainable=True)
        # Create a trainable weight variable for this layer.
        self.trainable_weights = [self.a, self.b]
        super(SigLayer, self).build(input_shape)  # built=true

    def call(self, x):
        # 1/(1 + exp(-ax + b))
        return K.pow(1+K.exp(-K.dot(x, self.a) + self.b), -1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)