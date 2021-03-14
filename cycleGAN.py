import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.layers import Input,Conv2D,Conv2DTranspose,Activation,Concatenate,Dropout,LeakyReLU,BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal


# A -> B的单向模型，包含了cycle loss, 对抗损失, identity loss, forward loss
def define_cycleganX2X(generatorA2B, discriminatorB, generatorB2A, img_shape):
    generatorA2B.trainable = True
    generatorB2A.trainable = False
    discriminatorB.trainable = False
    #只训练A->B的生成器

    inputA = Input(shape=img_shape)
    AgenB = generatorA2B(inputA)
    discriminateA2B = discriminatorB(AgenB)
    #对抗损失

    inputB = Input(shape=img_shape)
    genB2B = generatorA2B(inputB)
    #identity loss

    AgenBgenA = generatorB2A(AgenB)
    #forward loss

    BgenA = generatorB2A(inputB)
    BgenAgenB = generatorA2B(BgenA)
    #backward loss

    model = Model([inputA, inputB], [discriminateA2B, genB2B, AgenBgenA, BgenAgenB])

    opt = Adam(lr=0.0002, beta_1=0.5)

    model.compile(loss=['mse', 'mae', 'mae', 'mae'],
                  loss_weights=[1, 5, 10, 10],
                  optimizer=opt)

    return model

if __name__ == '__main__':
    A2B = define_cycleganX2X()


