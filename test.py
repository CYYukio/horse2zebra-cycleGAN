from os import listdir
import numpy as np
from numpy import asarray
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import tensorflow_addons as tfa

def load_images(path, size=(256, 256, 3)):
    data_list = list()
    # enumerate filenames in directory, assume all are images
    for filename in listdir(path):
        # load and resize the image
        pixels = load_img(path + filename, target_size=size)
        # convert to numpy array
        pixels = img_to_array(pixels)
        # store
        data_list.append(pixels)
    return asarray(data_list)

def save_data():
    path = './horse2zebra/'
    # load dataset A
    dataA = load_images(path + 'testA/')
    dataB = load_images(path + 'testB/')
    print("loaded horse", dataA.shape)
    print("loaded zebra", dataB.shape)

    filename = 'test.npz'
    np.savez_compressed(filename, dataA, dataB)
    print('Saved dataset: ', filename)



if __name__ == '__main__':
    data = np.load('test.npz')
    horse, zebra = data['arr_0'], data['arr_1']

    idx = np.random.randint(0, horse.shape[0], 1)
    horse_in = (horse[idx]-127.5)/127.5
    zebra_in = (zebra[idx]-127.5)/127.5

    generatorA2B = keras.models.load_model('g_model_A2B.h5')
    generatorB2A = keras.models.load_model('g_model_B2A.h5')
    genzebra = generatorA2B.predict(horse_in)
    genhorse = generatorB2A.predict(zebra_in)

    horse_src = (horse_in +1)/2.0
    genzebra = (genzebra + 1) / 2.0

    zebra_src = (zebra_in + 1) / 2.0
    genhorse = (genhorse + 1) / 2.0

    plt.subplot(2,2,1)
    plt.title('horse')
    plt.imshow(horse_src[0])
    plt.subplot(2, 2, 2)
    plt.title('gen-zebra')
    plt.imshow(genzebra[0])

    plt.subplot(2, 2, 3)
    plt.title('zebra')
    plt.imshow(zebra_src[0])
    plt.subplot(2, 2, 4)
    plt.title('gen-horse')
    plt.imshow(genhorse[0])

    plt.show()
