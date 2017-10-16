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
        w *= K.cast(K.greater(w, 0.), K.floatx())
        return w

class MaxNorm(Constraint):
    """MaxNorm weight constraint.
    Constrains the weights incident to each hidden unit
    to have a norm less than or equal to a desired value.
    # Arguments
        m: the maximum norm for the incoming weights.
        axis: integer, axis along which to calculate weight norms.
            For instance, in a `Dense` layer the weight matrix
            has shape `(input_dim, output_dim)`,
            set `axis` to `0` to constrain each weight vector
            of length `(input_dim,)`.
            In a `Conv2D` layer with `data_format="channels_last"`,
            the weight tensor has shape
            `(rows, cols, input_depth, output_depth)`,
            set `axis` to `[0, 1, 2]`
            to constrain the weights of each filter tensor of size
            `(rows, cols, input_depth)`.
    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting Srivastava, Hinton, et al. 2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
    """

    def __init__(self, max_value=1, axis=0):
        self.max_value = max_value
        self.axis = axis

    def __call__(self, w):
        norms = K.sqrt(K.sum(K.square(w), axis=self.axis, keepdims=True))
        desired = K.clip(norms, 0, self.max_value)
        w *= (desired / (K.epsilon() + norms))
        return w

    def get_config(self):
        return {'max_value': self.max_value,
                'axis': self.axis}