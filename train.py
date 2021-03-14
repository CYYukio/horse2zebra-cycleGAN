import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.layers import Input,Conv2D,Conv2DTranspose,Activation,Concatenate,Dropout,LeakyReLU,BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from generator import define_generator
from discriminator import define_discriminator
from cycleGAN import define_cycleganX2X
def load_data(filename):
    data = np.load('horse&zebra.npz')
    dataA, dataB = data['arr_0'], data['arr_1']

    dataA = (dataA - 127.5) / 127.5
    dataB = (dataB - 127.5) / 127.5

    return [dataA,dataB]

def real_samples(data, num, patchsize):
    idx = np.random.randint(0,data.shape[0], num)
    x = data[idx]
    y = np.ones((num,patchsize,patchsize,1))
    return x,y

def fake_samples(data, generator, patchsize):
    x = generator.predict(data)
    y = np.zeros((len(x),patchsize,patchsize,1))

    return x,y

def train(discriminatorA,
          discriminatorB,
          generatorA2B,
          generatorB2A,
          cycleganA2B,
          cycleganB2A,
          dataset):
    epochs, batchs = 10, 1

    patch_size = 16 #判别器生成的patch

    trainA, trainB = dataset

    bat_per_epo = int(len(trainA) / batchs)

    steps = bat_per_epo*epochs

    for i in range(steps):
        ARealX, ARealY = real_samples(trainA, batchs, patch_size)
        BRealX, BRealY = real_samples(trainB, batchs, patch_size)

        AgenBX, AgenBY = fake_samples(ARealX,generatorA2B,patch_size)
        BgenAX, BgenAY = fake_samples(BRealX,generatorB2A,patch_size)


        #A->B 生成器loss
        genlossA2B, _, _, _, _ = cycleganA2B.train_on_batch([ARealX, BRealX], [BRealY, BRealX, ARealX, BRealX])

        #A->B判别器
        dis_lossBReal = discriminatorB.train_on_batch(BRealX, BRealY)
        dis_lossBFake = discriminatorB.train_on_batch(AgenBX, AgenBY)

        # B->A 生成器loss
        genlossB2A, _, _, _, _ = cycleganB2A.train_on_batch([BRealX, ARealX], [ARealY, ARealX, BRealX, ARealX])

        #B->A判别器
        dis_lossAReal = discriminatorA.train_on_batch(ARealX, ARealY)
        dis_lossAFake = discriminatorA.train_on_batch(BgenAX, BgenAY)

        if (i + 1) % (bat_per_epo) == 0:
            print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i + 1, dis_lossAReal, dis_lossAFake, dis_lossBReal, dis_lossBFake, generatorA2B, genlossB2A))

        generatorA2B.save('generatorA2B.h5')
        generatorB2A.save('generatorB2A.h5')

if __name__ == '__main__':
    dataset = load_data('horse&zebra.npz')
    print('Loaded', dataset[0].shape, dataset[1].shape)

    image_shape = (256,256,3)

    generatorA2B = define_generator(image_shape)
    generatorB2A = define_generator(image_shape)
    discriminatorA = define_discriminator(image_shape)
    discriminatorB = define_discriminator(image_shape)

    cycleganA2B = define_cycleganX2X(generatorA2B,discriminatorB,generatorB2A,image_shape)
    cycleganB2A = define_cycleganX2X(generatorB2A, discriminatorA, generatorA2B, image_shape)

    train(discriminatorA=discriminatorA,
          discriminatorB=discriminatorB,
          generatorA2B=generatorA2B,
          generatorB2A=generatorB2A,
          cycleganA2B=cycleganA2B,
          cycleganB2A=cycleganB2A,
          dataset=dataset)