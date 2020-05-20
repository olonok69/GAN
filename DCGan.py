# -*- coding: utf-8 -*-
"""
    Program: DCGAN,s examples:-D
    Module : DCGAB.py
    Descr  : Deep Convolutional Generative Adversarial network
    Date   : Spring 2020 (MAY)
"""
__author__ = 'Juan Huertas'
__email__ = 'olonok@gmail.com'
__version__ = '1.0'
__status__ = 'Production'

# Needed system modules

from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from tensorflow.compat.v1.keras.datasets.cifar10 import load_data
from tensorflow.compat.v1.keras.optimizers import Adam
from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import Dense
from tensorflow.compat.v1.keras.layers import Reshape
from tensorflow.compat.v1.keras.layers import Flatten
from tensorflow.compat.v1.keras.layers import Conv2D
from tensorflow.compat.v1.keras.layers import Conv2DTranspose
from tensorflow.compat.v1.keras.layers import LeakyReLU
from tensorflow.compat.v1.keras.layers import Dropout
from tensorflow.compat.v1.keras.models import load_model
import random
from numpy import asarray
from matplotlib import pyplot
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

class dcgan_class(object):

    def __init__(self, log_obj, output_dir, pic_dir, models_dir, latent_dim):

        self.log_obj = log_obj # logging directory
        self.output_dir=output_dir #
        self.pic_dir=pic_dir # directory to save intermediary pics
        self.models_dir=models_dir # directory to save model
        self.latent_dim=latent_dim # latent vector dimesion
        self.model=None
        self.session=None
        self.model_fil=None

        return

    def load_model(self, model_file):
        self.session = tf.Session()
        tf.compat.v1.keras.backend.set_session(self.session)
        self.model_file=model_file
        self.model=load_model(model_file)
        return self.model
    def predict(self):
        # generate a random vector 100 dim latent space
        vector = asarray([[random.uniform(0, 1) for _ in range(100)]])
        # load model
        self.session = tf.Session()
        tf.compat.v1.keras.backend.set_session(self.session)
        self.model = self.load_model(self.model_file)

        with self.session.as_default():
            with self.session.graph.as_default():
                X = self.model.predict(vector)
                # scale from [-1,1] to [0,1]
                X = (X + 1) / 2.0
        return X

    @staticmethod
    # define the standalone discriminator model
    def define_discriminator(in_shape=(32, 32, 3)):
        model = Sequential()
        # normal
        model.add(Conv2D(64, (3, 3), padding='same', input_shape=in_shape))
        model.add(LeakyReLU(alpha=0.2))
        # downsample
        model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # downsample
        model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # downsample
        model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # classifier
        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    @staticmethod
    # define the standalone generator model
    def define_generator(latent_dim):
        model = Sequential()
        # foundation for 4x4 image
        n_nodes = 256 * 4 * 4
        model.add(Dense(n_nodes, input_dim=latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((4, 4, 256)))
        # upsample to 8x8
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # upsample to 16x16
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # upsample to 32x32
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # output layer
        model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))
        return model

    @staticmethod
    # define the combined generator and discriminator model, for updating the generator
    def define_gan(g_model, d_model):
        # make weights in the discriminator not trainable
        d_model.trainable = False
        # connect them
        model = Sequential()
        # add generator
        model.add(g_model)
        # add the discriminator
        model.add(d_model)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

    @staticmethod
    # load and prepare cifar10 training images
    def load_real_samples():
        # load cifar10 dataset
        (trainX, _), (_, _) = load_data()
        # convert from unsigned ints to floats
        X = trainX.astype('float32')
        # scale from [0,255] to [-1,1]
        X = (X - 127.5) / 127.5
        return X

    @staticmethod
    # select real samples
    def generate_real_samples(dataset, n_samples):
        # choose random instances
        ix = randint(0, dataset.shape[0], n_samples)
        # retrieve selected images
        X = dataset[ix]
        # generate 'real' class labels (1)
        y = ones((n_samples, 1))
        return X, y

    @staticmethod
    # generate points in latent space as input for the generator
    def generate_latent_points(latent_dim, n_samples):
        # generate points in the latent space
        x_input = randn(latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(n_samples, latent_dim)
        return x_input

    @staticmethod
    # use the generator to generate n fake examples, with class labels
    def generate_fake_samples(g_model, latent_dim, n_samples):
        # generate points in latent space
        x_input = dcgan_class.generate_latent_points(latent_dim, n_samples)
        # predict outputs
        X = g_model.predict(x_input)
        # create 'fake' class labels (0)
        y = zeros((n_samples, 1))
        return X, y

    @staticmethod
    # create and save a plot of generated images
    def save_plot(examples, epoch, n=7):
        # scale from [-1,1] to [0,1]
        examples = (examples + 1) / 2.0
        # plot images
        for i in range(n * n):
            # define subplot
            pyplot.subplot(n, n, 1 + i)
            # turn off axis
            pyplot.axis('off')
            # plot raw pixel data
            pyplot.imshow(examples[i])
        # save plot to file
        filename = './pics/generated_plot_e%03d.png' % (epoch + 1)
        pyplot.savefig(filename)
        pyplot.close()

    @staticmethod
    # evaluate the discriminator, plot generated images, save generator model
    def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=150):
        # prepare real samples
        X_real, y_real = dcgan_class.generate_real_samples(dataset, n_samples)
        # evaluate discriminator on real examples
        _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
        # prepare fake examples
        x_fake, y_fake = dcgan_class.generate_fake_samples(g_model, latent_dim, n_samples)
        # evaluate discriminator on fake examples
        _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
        # summarize discriminator performance
        print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real * 100, acc_fake * 100))
        # save plot
        dcgan_class.save_plot(x_fake, epoch)
        # save the generator model tile file
        filename = './models/generator_model_%03d.h5' % (epoch + 1)
        g_model.save(filename)

    @staticmethod
    # train the generator and discriminator
    def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=50, n_batch=256):
        bat_per_epo = int(dataset.shape[0] / n_batch)
        half_batch = int(n_batch / 2)
        # manually enumerate epochs
        for i in range(n_epochs):
            # enumerate batches over the training set
            for j in range(bat_per_epo):
                # get randomly selected 'real' samples
                X_real, y_real = dcgan_class.generate_real_samples(dataset, half_batch)
                # update discriminator model weights
                d_loss1, _ = d_model.train_on_batch(X_real, y_real)
                # generate 'fake' examples
                X_fake, y_fake = dcgan_class.generate_fake_samples(g_model, latent_dim, half_batch)
                # update discriminator model weights
                d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
                # prepare points in latent space as input for the generator
                X_gan = dcgan_class.generate_latent_points(latent_dim, n_batch)
                # create inverted labels for the fake samples
                y_gan = ones((n_batch, 1))
                # update the generator via the discriminator's error
                g_loss = gan_model.train_on_batch(X_gan, y_gan)
                # summarize loss on this batch
                print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                      (i + 1, j + 1, bat_per_epo, d_loss1, d_loss2, g_loss))
            # evaluate the model performance, sometimes
            if (i + 1) % 10 == 0:
                dcgan_class.summarize_performance(i, g_model, d_model, dataset, latent_dim)