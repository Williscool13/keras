from .layers import Layer
from .layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D
from .layers.merge import Concatenate

class Inception(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Inception, self).__init__(**kwargs)
        self.layer1a = Conv2D(10, (1,1), padding='same', activation='relu')
        self.layer1b = Conv2D(10, (3,3), padding='same', activation='relu')
        self.layer2a = Conv2D(10, (1,1), padding='same', activation='relu')
        self.layer2b = Conv2D(10, (5,5), padding='same', activation='relu')
        self.layer3a = MaxPooling2D((3,3), strides=(1,1), padding='same')
        self.layer3b = Conv2D(10, (1,1), padding='same', activation='relu')

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], input_shape[2], input_shape[3]),
                                      initializer='uniform',
                                      trainable=True)
        super(Inception, self).build(input_shape)


    def call(self, x):
        a = self.layer1a(x)
        a = self.layer1b(a)
        b = self.layer2a(x)
        b = self.layer2b(b)
        c = self.layer3a(x)
        c = self.layer3b(c)
        return Concatenate([a,b,c])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3])
