from os import listdir
import numpy as np
from numpy import asarray
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
# load all images in a directory into memory
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
    dataA = load_images(path + 'trainA/')
    dataB = load_images(path + 'trainB/')
    print("loaded horse", dataA.shape)
    print("loaded zebra", dataB.shape)

    filename = 'horse&zebra.npz'
    np.savez_compressed(filename, dataA, dataB)
    print('Saved dataset: ', filename)

if __name__ == '__main__':
    #save_data()

    data = np.load('horse&zebra.npz')
    dataA, dataB = data['arr_0'], data['arr_1']
    print('Loaded: ', dataA.shape, dataB.shape)
    # plot source images
    n_samples = 3
    for i in range(n_samples):
        plt.subplot(2, n_samples, 1 + i)
        plt.axis('off')
        plt.imshow(dataA[i].astype('uint8'))
    # plot target image
    for i in range(n_samples):
        plt.subplot(2, n_samples, 1 + n_samples + i)
        plt.axis('off')
        plt.imshow(dataB[i].astype('uint8'))
    plt.show()


