from keras import backend as K
from keras.layers import merge, Activation
from keras.models import Model
from keras.engine.topology import Layer
import keras.initializers
import numpy as np

'''All functions and classes related to adding the 3 custom layers after the score output'''

def add_custom(F):
    ''' F&G concatenated
        3 parallel networks, with the same F and different Gs'''
    shared_input = F.input
    f_out = F.output
    # Create G, higher score is plus
    processed_norm = InvSigLayer()(f_out)
    processed_pre = GausLayer()(f_out)
    processed_plus = SigLayer()(f_out)
    # Add 3 dimensional softmax output
    merge_out = merge([processed_plus, processed_pre, processed_norm], mode='concat')
    activ_out = Activation('softmax')(merge_out)
    # Create the whole network
    concat_abs_net = Model(shared_input, activ_out)
    return concat_abs_net

def MLloss(y_true, y_pred):
    """
    Negative log likelihood of absolute model
    y_true = [100],[010],[001]
    y_pred = soft[g_1(s), g_2(s), g_3(s)]
    Take the g output corresponding to the label
    """
    diff = K.dot(y_pred, K.transpose(y_true))
    return -K.log(diff)

def LogisticLoss(y_true, y_pred):
    """
    Negative log likelihood of Bradley-Terry Penalty, to be minimized. y = beta.*x
    """
    exponent = K.exp(-y_true * (y_pred))
    return K.log(1 + exponent)

class GausLayer(Layer):
    def __init__(self, input_dim = 1, output_dim=1, **kwargs):
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
                     / (2 *self.var)) \
                    * 1. / (K.sqrt(2 * np.pi * self.var) + K.epsilon())

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

class InvSigLayer(Layer):
    def __init__(self, input_dim = 1, output_dim=1, **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        super(InvSigLayer, self).__init__(**kwargs)

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
        super(InvSigLayer, self).build(input_shape)  # built=true

    def call(self, x):
        # 1 - 1/(1 + exp(a(x - b)))
        # Dot: matrix multiplication & repeat weights for batch size adaptation
        # return 1 - 1. / (1+K.exp(K.dot((x - K.repeat_elements(self.b, K.shape(x)[0], axis=0)), self.a)))
        return 1 - 1. / (1+K.exp(K.dot((x - self.b), self.a)))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

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

class Constraint(object):

    def __call__(self, w):
        return w

    def get_config(self):
        return {}

class PosVal(Constraint):
    """Constrains the weights to be positive.
    """

    def __call__(self, w):
        w *= K.cast(K.greater(w, 0.), K.floatx())
        return w