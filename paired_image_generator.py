import numpy as np
import random
import os
import cv2
import tensorflow as tf
import math

class PairedImageGenerator2(object):
    'Generates data for Keras'
    def __init__(self, dim=(256, 256), n_channels=1, batch_size=16, images_path_cover='/', images_path_stego='/', shuffle=True, augment=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.images_path_stego = images_path_stego
        self.images_path_cover = images_path_cover
        self.shuffle = shuffle
        self.augment = augment
        
        # cover and stego filenames should be the same
        self.image_names = os.listdir(self.images_path_cover)

        self.shuffle_images()
    
    '''
    Goes through the dataset and outputs one batch at a time.
    ''' 
    def generate(self):
        'Generates batches of samples'

        # Infinite loop
        while 1:
            # Randomize the filenames
            self.shuffle_images()

            # Number of batches to complete the whole data set.
            imax = int(len(self.image_names)/(self.batch_size//2))
            
            for i in range(imax):

                print(i)
                # Select the 
                batch_filenames = self.image_names[i*(self.batch_size//2):(i+1)*(self.batch_size//2)]

                # Generate data
                X, y = self.__data_generation(batch_filenames)

                yield X, y
    
    def data_augmentation(self, image):
        # Rotate the image 90, -90 or 0 degrees.
        image = np.rot90(image, np.random.choice([-1, 0, 1]))
        
        # With 50% probability, the image is flipped horizontally.
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
    
        return image
    
    def __data_generation(self, batch_filenames):
        'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)

        # Initialization

        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        y = np.empty((self.batch_size), dtype = int)

        # Generate data

        for i, filename in enumerate(batch_filenames):

            # Read both the cover and the stego images
            im_cover = cv2.imread(os.path.join(self.images_path_cover, filename), cv2.IMREAD_UNCHANGED)
            im_cover = im_cover.astype(np.single).reshape(self.dim[0], self.dim[1], self.n_channels)
        
            im_stego = cv2.imread(os.path.join(self.images_path_stego, filename), cv2.IMREAD_UNCHANGED)
            im_stego = im_stego.astype(np.single).reshape(self.dim[0], self.dim[1], self.n_channels)

            # Check whether Data Augmentation must be used
            if self.augment:
                im_cover = self.data_augmentation(im_cover)
                im_stego = self.data_augmentation(im_stego)

            # Add the images
            X[i*2, :, :, :] = im_cover
            y[i*2] = 0
            X[i*2+1, :, :, :] = im_stego
            y[i*2+1] = 1

        return X, y


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        print("EPOCH FINISHED")

    
    def shuffle_images(self):
        'Shuffle filenames at the end of each epoch if shuffle == True'
        if self.shuffle == True:
            print("SHUFFLING IMAGES")
            random.shuffle(self.image_names)


class PairedImageGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dim=(256, 256), n_channels=1, batch_size=16, images_path_cover='/', images_path_stego='/', shuffle=True, augment=True, seed=None):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.images_path_stego = images_path_stego
        self.images_path_cover = images_path_cover
        self.shuffle = shuffle
        self.augment = augment

        # Set the seed
        if seed != None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
        
        # cover and stego filenames should be the same
        self.image_names = os.listdir(self.images_path_cover)
        self.indices = np.arange(len(self.image_names))

        # Shuffle the indices
        self.shuffle_indices()


    def __getitem__(self, i):
        # Select the indices that should be used now
        inds = self.indices[i * (self.batch_size//2):(i + 1) * (self.batch_size//2)]

        # Select the filenames
        batch_filenames = [self.image_names[i] for i in inds]

        # Get the X and the y
        X, y = self.__data_generation(batch_filenames)

        return X, y
    

    def __len__(self):
        return math.ceil(len(self.image_names) / (self.batch_size//2))


    def on_epoch_end(self):
        self.shuffle_indices()


    def preprocessing(self, image):
        image = image * 1./255
        return image


    def data_augmentation(self, cover, stego):
        rotation = np.random.choice([-1, 0, 1])
        # Rotate the image 90, -90 or 0 degrees.
        cover = np.rot90(cover, rotation)
        stego = np.rot90(stego, rotation)
        
        # With 50% probability, the image is flipped horizontally.
        if tf.random.uniform(()) > 0.5:
            cover = tf.image.flip_left_right(cover)
            stego = tf.image.flip_left_right(stego)
    
        return cover, stego
    

    def __data_generation(self, batch_filenames):
        'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)

        # Initialization

        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        y = np.empty((self.batch_size), dtype = int)

        # Generate data

        for i, filename in enumerate(batch_filenames):

            # Read both the cover and the stego images
            im_cover = cv2.imread(os.path.join(self.images_path_cover, filename), cv2.IMREAD_UNCHANGED)
            im_cover = im_cover.astype(np.single).reshape(self.dim[0], self.dim[1], self.n_channels)
        
            im_stego = cv2.imread(os.path.join(self.images_path_stego, filename), cv2.IMREAD_UNCHANGED)
            im_stego = im_stego.astype(np.single).reshape(self.dim[0], self.dim[1], self.n_channels)

            # Preprocess the images (Scale by 255)
            im_cover = self.preprocessing(im_cover)
            im_stego = self.preprocessing(im_stego)

            # Check whether Data Augmentation must be used
            if self.augment:
                im_cover, im_stego = self.data_augmentation(im_cover, im_stego)

            # Add the images
            X[i*2, :, :, :] = im_cover
            y[i*2] = 0
            X[i*2+1, :, :, :] = im_stego
            y[i*2+1] = 1

        return X, y
    

    def shuffle_indices(self):
        'Shuffle filenames at the end of each epoch if shuffle == True'
        if self.shuffle == True:
            np.random.shuffle(self.indices)