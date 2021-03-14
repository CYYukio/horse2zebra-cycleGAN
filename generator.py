import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.layers import Input,Conv2D,Conv2DTranspose,Activation,Concatenate,Dropout,LeakyReLU,BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
import tensorflow_addons as tfa
#中间层用resnet
def resnet_layer(filtersnum, input):
    init = RandomNormal(stddev=0.02)
    layer = Conv2D(filters=filtersnum,
                   kernel_size=(3, 3),
                   padding='same',
                   kernel_initializer=init)(input)
    layer = tfa.layers.InstanceNormalization(axis=-1)(layer)
    layer = Activation('relu')(layer)

    layer = Conv2D(filters=filtersnum,
                   kernel_size=(3, 3),
                   padding='same',
                   kernel_initializer=init)(layer)
    layer = tfa.layers.InstanceNormalization(axis=-1)(layer)
    output = Concatenate()([layer, input])

    return output


def define_generator(img_shape):
    init = RandomNormal(stddev=0.02)
    input = Input(shape=img_shape)

    down_layer = Conv2D(filters=64, kernel_size=(7,7), kernel_initializer=init, padding='same')(input)  #(256,256,3)
    down_layer = tfa.layers.InstanceNormalization(axis=-1)(down_layer)
    down_layer = Activation('relu')(down_layer)

    down_layer = Conv2D(filters=128, kernel_size=(3,3), strides=(2, 2),kernel_initializer=init, padding='same')(down_layer) #(128,128,128)
    down_layer = tfa.layers.InstanceNormalization(axis=-1)(down_layer)
    down_layer = Activation('relu')(down_layer)

    down_layer = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), kernel_initializer=init, padding='same')(down_layer)  # (64,64,256)
    down_layer = tfa.layers.InstanceNormalization(axis=-1)(down_layer)
    down_layer = Activation('relu')(down_layer)


    mid_layer = resnet_layer(256,down_layer)
    mid_layer = resnet_layer(256,mid_layer)
    mid_layer = resnet_layer(256, mid_layer)
    mid_layer = resnet_layer(256, mid_layer)

    #(64,64,*)
    up_layer = Conv2DTranspose(256, kernel_size=(3, 3), strides=(2, 2), kernel_initializer=init, padding='same')(mid_layer) #(128,128,256)
    up_layer = tfa.layers.InstanceNormalization(axis=-1)(up_layer)
    up_layer = Activation('relu')(up_layer)

    up_layer = Conv2DTranspose(256, kernel_size=(3, 3), strides=(2, 2), kernel_initializer=init, padding='same')(up_layer) #(256,256,512)
    up_layer = tfa.layers.InstanceNormalization(axis=-1)(up_layer)
    up_layer = Activation('relu')(up_layer)

    output = Conv2D(3, kernel_size=(7,7), kernel_initializer=init, padding='same')(up_layer) #(256,256,3)
    output = tfa.layers.InstanceNormalization(axis=-1)(output)
    output = Activation('tanh')(output)  #尽量往-1或1激活

    model = Model(input, output)

    return model

if __name__ == '__main__':
    generator = define_generator((256,256,3))
    generator.summary()
