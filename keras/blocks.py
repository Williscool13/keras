from .layers import Layer, LeakyReLU, BatchNormalization
from .layers.pooling import GlobalAveragePooling2D
from .layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D, DepthwiseConv2D
from .layers.merge import Concatenate, Add, Multiply
from .layers.core import Dense

class Inception(Layer):

    def __init__(self, **kwargs):
        super(Inception, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel1',
                                      shape=(input_shape[1], input_shape[2], input_shape[3]),
                                      initializer='uniform',
                                      trainable=True)
        super(Inception, self).build(input_shape)


    def call(self, x):
        h,w,c = x.shape[1:]
        conv1 = Conv2D(c, (1,1), padding='same', activation='relu')(x)
        # 3x3 conv
        conv3 = Conv2D(c, (1,1), padding='same', activation='relu')(x)
        conv3 = Conv2D(c, (3,3), padding='same', activation='relu')(conv3)
        # 5x5 conv
        conv5 = Conv2D(c, (1,1), padding='same', activation='relu')(x)
        conv5 = Conv2D(c, (5,5), padding='same', activation='relu')(conv5)
        # 3x3 max pooling
        pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
        pool = Conv2D(c, (1,1), padding='same', activation='relu')(pool)
        # concatenate filters, assumes filters/channels last
        layer_out = Concatenate(axis=-1)([conv1, conv3, conv5, pool])

        return layer_out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3])


class Residual(Layer):
    def __init__(self, **kwargs):
        super(Residual, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel2',
                                    shape=(input_shape[1], input_shape[2], input_shape[3]),
                                    initializer='uniform',
                                    trainable=True)
        super(Residual, self).build(input_shape)

    def call(self, x):
        h,w,c = x.shape[1:]
        a = Conv2D(c, 3, strides=1, padding='same')(x)
        a = BatchNormalization()(a)
        a = LeakyReLU()(a)

        a = Conv2D(c, 3, strides=1, padding='same')(a)
        a = BatchNormalization()(a)

        b = Conv2D(c, 1, strides=1, padding='same')(x)
        b = BatchNormalization()(b)

        a = Add()([b, a])
        a = LeakyReLU()(a)

        return a

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3])


class VGG(Layer):
    def __init__(self, **kwargs):
        super(VGG, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel3',
                                    shape=(input_shape[1], input_shape[2], input_shape[3]),
                                    initializer='uniform',
                                    trainable=True)
        super(VGG, self).build(input_shape)

    def call(self, x):
        h,w,c = x.shape[1:]
        for _ in range(3):
            x = Conv2D(c * 2, (3,3), padding='same', activation='relu')(x)
        # add max pooling layer
        out = MaxPooling2D((2,2), strides=(2,2))(x)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3])


class SqueezeExcite(Layer):
    def __init__(self, **kwargs):
        super(SqueezeExcite, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel4',
                                    shape=(input_shape[1], input_shape[2], input_shape[3]),
                                    initializer='uniform',
                                    trainable=True)
        super(SqueezeExcite, self).build(input_shape)

    def call(self, x):
        h,w,c = x.shape[1:]
        #ratio recommended to be 6, can be hyperparameterized
        ratio = 16
        x1 = GlobalAveragePooling2D()(x)
        x2 = Dense(c // ratio, activation='relu')(x1)
        x3 = Dense(c, activation='sigmoid')(x2)

        out = Multiply()([x, x3])

        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3])



class InvertedResidual(Layer):
    """
    Implementation of Inverted Residual as seen in Google's MobileNet. 
    This implementation is a reverse bottleneck along with a residual
    """
    def __init__(self, **kwargs):
        super(InvertedResidual, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel5',
                                    shape=(input_shape[1], input_shape[2], input_shape[3]),
                                    initializer='uniform',
                                    trainable=True)
        super(InvertedResidual, self).build(input_shape)

    def call(self, x):
        #hyperparameterized by squeeze and excite channel shape
        h,w,c = x.shape[1:]
        squeeze = c
        expand = c * 4

        m = Conv2D(expand, (1,1), activation='relu')(x)
        m = DepthwiseConv2D((3,3), activation='relu', padding='same')(m)
        m = Conv2D(squeeze, (1,1), activation='relu')(m)

        return Add()([m, x])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3])
