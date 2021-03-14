import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input,Conv2D,Conv2DTranspose,Activation,Concatenate,Dropout,LeakyReLU,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import tensorflow_addons as tfa

def define_discriminator(img_shape):
    init = RandomNormal(stddev=0.02)

    in_image = Input(shape=img_shape)

    layer = Conv2D(filters=64,
                   kernel_size=(4, 4),
                   strides=(2, 2),
                   padding='same',
                   kernel_initializer=init)(in_image) #(128,128,64)
    layer = LeakyReLU(alpha=0.2)(layer)

    layer = Conv2D(filters=128,
                   kernel_size=(4, 4),
                   strides=(2, 2),
                   padding='same',
                   kernel_initializer=init)(layer) #(64,64,128)
    layer = tfa.layers.InstanceNormalization(axis=-1)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)

    layer = Conv2D(filters=256,
                   kernel_size=(4, 4),
                   strides=(2, 2),
                   padding='same',
                   kernel_initializer=init)(layer)  # (32,32,256)
    layer = tfa.layers.InstanceNormalization(axis=-1)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)

    layer = Conv2D(filters=512,
                   kernel_size=(4, 4),
                   strides=(2, 2),
                   padding='same',
                   kernel_initializer=init)(layer)  # (16,16,512)
    layer = tfa.layers.InstanceNormalization(axis=-1)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)

    layer = Conv2D(filters=512,
                   kernel_size=(4, 4),
                   padding='same',
                   kernel_initializer=init)(layer)  # (16,16,512)
    layer = tfa.layers.InstanceNormalization(axis=-1)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)

    patch_out = Conv2D(filters=1,
                       kernel_size=(4, 4),
                       padding='same',
                       kernel_initializer=init)(layer) # (16,16,1)

    model = Model(in_image,patch_out)

    model.compile(loss='mse',
                  optimizer=Adam(lr=0.0002,beta_1=0.5),
                  loss_weights=[0.5])

    return model

if __name__ == '__main__':
    d = define_discriminator((256,256,3))
    d.summary()


