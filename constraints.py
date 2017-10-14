from __future__ import absolute_import
from keras import backend as K

class Constraint(object):

    def __call__(self, w):
        return w

    def get_config(self):
        return {}

class PosVal(Constraint):
    """Constrains the weights to be positive.
    """

    def __call__(self, w):
        w *= K.cast(K.greater(w, K.epsilon()), K.floatx())
        return w