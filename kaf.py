import tensorflow as tf
import numpy as np 


def dictionaryGen(D):
    
    """ Generate and return 
        - D integers on the x axis evenly distributed around 0
        - The step size Δx """

    d_pos = np.linspace(-10, 10, num= D, retstep=True, dtype=np.float32)
    return (d_pos[1], d_pos[0])


def kafActivation(x, a, d, k_bwidth):

    """ For each element in x, compute the weighted sum of 1D Gaussian kernel as defined in arXiv:1707.04035 """
    
    x = tf.math.square(x - d)
    x = a * tf.math.exp((-k_bwidth) * x)

    return tf.reduce_sum(x, -1)



from tensorflow.keras.layers import Layer

class Kaf(Layer):

    """ Kernel Activation Function implemented as a keras layer to allow parameters learning 
    
        It takes as input D which is supposed to be the size of the dictionary
        and the label conv to indicate if the activation comes from a convolutional layer
        
        In particular, there are two supported activations only:
        - A batch of flattened units i.e. of shape = (b, x)
        - A batch of 2DConvolutions i.e. of shape = (b, x, y, f) where f is supposed to be the channels size
        
        If the shape does not match with any of the latter, an error is thrown 
        
        References
        ----------
    [1] Scardapane, S., Van Vaerenbergh, S., Totaro, S. and Uncini, A., 2019. 
        Kafnets: kernel-based non-parametric activation functions for neural networks. 
        Neural Networks, 110, pp. 19-32.
    [2] Marra, G., Zanca, D., Betti, A. and Gori, M., 2018. 
        Learning Neuron Non-Linearities with Kernel-Based Deep Neural Networks. 
        arXiv preprint arXiv:1807.06302."""

    def __init__(self, D, conv=False, **kwargs):

        super(Kaf, self).__init__()
        
        # Init constants
        self.D = D
        self.conv = conv
        
        step, dict = dictionaryGen(D)
        self.d = tf.stack(dict)
        self.k_bandw = 1/(6*(np.power(step,2)))
        

    def build(self, input_shape):

        # Raise an exception if the input rank is not white listed
        try:
            input_shape.assert_has_rank(2)
        except ValueError:
            try:
                input_shape.assert_has_rank(4)
            except ValueError:
                raise ValueError("The input shape for Kaf must be either a dense batch (b,x) \n or a gridlike batch (b, x, y, f)")

        # Init mix coefficients
        self.a = self.add_weight(shape=(1, input_shape[-1], self.D),   
                                 initializer= 'random_normal', 
                                 trainable=True) 
        
        # Adjust dimensions in order to exploit broadcasting and compute the entire batch all at once
        if not self.conv:
            self.d = tf.reshape(self.d, shape=(1, 1, self.D))
        
        else:
            self.a = tf.reshape(self.a, shape=(1,1,1,-1,self.D))
            self.d = tf.reshape(self.d, shape=(1, 1, 1, 1, self.D))
            

   
    def call(self, inputs):
        
        inputs = tf.expand_dims(inputs, -1)
        return kafActivation(inputs, self.a, self.d, self.k_bandw) 
                 