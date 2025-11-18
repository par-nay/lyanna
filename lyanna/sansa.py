#! /usr/bin/env python

import tensorflow as tf 
from lyanna.datautils import *
from lyanna.utils import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

class ResBlock:
    def __init__(
            self, 
            num_filters, 
            kernel_size, 
            strides = 2, 
            activation = None, 
            padding = 'same', 
            regularizer = None, 
            initializer = None, 
            dropout = 0.0, 
            idx = 0
        ):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides     = strides 
        self.activation  = activation
        self.padding     = padding
        self.regularizer = regularizer
        self.initializer = initializer
        self.dropout     = dropout
        self.idx         = idx 
        self.num_clayers = 3
        
    def __call__(self, x):
        x = tf.keras.layers.Conv1D(
            self.num_filters, 
            self.kernel_size, 
            strides = self.strides, 
            activation = None, 
            padding = self.padding, 
            kernel_regularizer = self.regularizer, 
            kernel_initializer = self.initializer, 
            name = f'conv_{self.num_clayers*self.idx}',
        )(x)
        x_skip = tf.identity(x)
        x = tf.keras.layers.Conv1D(
            self.num_filters, 
            self.kernel_size, 
            strides = 1, 
            activation = self.activation, 
            padding = self.padding, 
            kernel_regularizer = self.regularizer, 
            kernel_initializer = self.initializer, 
            name = f'conv_{self.num_clayers*self.idx+1}',
        )(x)
        x = tf.keras.layers.Conv1D(
            self.num_filters, 
            self.kernel_size, 
            strides = 1, 
            activation = None, 
            padding = self.padding, 
            kernel_regularizer = self.regularizer, 
            kernel_initializer = self.initializer, 
            name = f'conv_{self.num_clayers*self.idx+2}',
        )(x)
        x = tf.keras.layers.Add(
            name = f'skip_{self.idx}'
        )([x_skip, x])
        x = tf.keras.layers.LeakyReLU()(x)
        return x
    

class ResBlockTranspose:
    def __init__(
        self, 
        num_filters, 
        kernel_size, 
        strides = 2, 
        activation = None, 
        padding = 'same', 
        regularizer = None, 
        initializer = None, 
        dropout = 0.0, 
        idx = 0
    ):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides     = strides 
        self.activation  = activation
        self.padding     = padding
        self.regularizer = regularizer
        self.initializer = initializer
        self.dropout     = dropout
        self.idx         = idx 
        self.num_clayers = 3
        
    def __call__(self, x):
        x = tf.keras.layers.Conv1DTranspose(
            self.num_filters, 
            self.kernel_size, 
            strides = self.strides, 
            activation = None, 
            padding = self.padding, 
            kernel_regularizer = self.regularizer, 
            kernel_initializer = self.initializer, 
            name = f'convT_{self.num_clayers*self.idx}',
        )(x)
        x_skip = x
        x = tf.keras.layers.Conv1DTranspose(
            self.num_filters, 
            self.kernel_size, 
            strides = 1, 
            activation = self.activation, 
            padding = self.padding, 
            kernel_regularizer = self.regularizer, 
            kernel_initializer = self.initializer, 
            name = f'convT_{self.num_clayers*self.idx+1}',
        )(x)
        x = tf.keras.layers.Conv1DTranspose(
            self.num_filters, 
            self.kernel_size, 
            strides = 1, 
            activation = None, 
            padding = self.padding, 
            kernel_regularizer = self.regularizer, 
            kernel_initializer = self.initializer, 
            name = f'convT_{self.num_clayers*self.idx+2}',
        )(x)
        x = tf.keras.layers.Add(
            name = f'dec_skip_{self.idx}'
        )([x_skip, x])
        x = tf.keras.layers.LeakyReLU()(x)
        return x
    


class Sansa(tf.keras.Model):
    def initialize_augmentations(
        self, 
        seed = 0, 
        p_flip = 0.5
    ):
        # self.Noise = Noise(CNR, v_h_skewer, CNR = True, N_pix_spectrum = 512)
        self.Augmentation = Augmentation(p_flip, seed = seed)
        
    def train_step(self, batch):
        inputs, labels = batch
        aug_inputs     = self.Augmentation.augment_without_noise(inputs)
        with tf.GradientTape() as tape:
            y_pred     = self(aug_inputs, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss       = self.compute_loss(y=labels, y_pred=y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(labels, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


class nSansa(tf.keras.Model):
    def initialize_augmentations(
        self, 
        p_flip, 
        v_h_skewer, 
        seed = 0, 
        SNR = 100, 
        CNR = True, 
        N_pix_spectrum = 256, 
        SNR_max = None,  
        mode = 'p0s0', 
        batch_size = None
    ):
        self.Augmentation = Augmentation(p_flip, seed = seed)
        self.Augmentation.init_noise(SNR, v_h_skewer, CNR = CNR, N_pix_spectrum = N_pix_spectrum, mode = mode, batch_size = batch_size, SNR_max = SNR_max)
        
    def train_step(self, batch):
        inputs, labels = batch
        # print(inputs.shape, labels.shape)
        if self.Augmentation.mode == 'p0s0':
            aug_inputs     = self.Augmentation.augment_with_noise(inputs)
        elif self.Augmentation.mode == 'p0s1':
            aug_inputs, _  = self.Augmentation.augment_with_noise(inputs)    
        with tf.GradientTape() as tape:
            y_pred     = self(aug_inputs, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss       = self.compute_loss(y=labels, y_pred=y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(labels, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    

class nSansaQuery(tf.keras.Model):
    # impl_0-
    def initialize_augmentations(
        self, 
        p_flip, 
        v_h_skewer, 
        seed = 0, 
        SNR = 20, 
        CNR = True, 
        N_pix_spectrum = 256, 
        mode = 'p0s1', 
        SNR_max = 100, 
        batch_size = None
    ):
        self.Augmentation = Augmentation(p_flip, seed = seed)
        self.Augmentation.init_noise(SNR, v_h_skewer, CNR = CNR, N_pix_spectrum = N_pix_spectrum, mode = mode, SNR_max = SNR_max, batch_size = batch_size)


    def train_step(self, batch):
        inputs, labels = batch
        # print(inputs.shape, labels.shape)
        noisy_inputs, sigmas = self.Augmentation.augment_with_noise(inputs)
        y_true = tf.concat([labels, tf.dtypes.cast(tf.reshape(sigmas, (tf.shape(labels)[-2],1)), tf.float32)], axis=-1)
        # print("y_true shape:", y_true.shape)
        with tf.GradientTape() as tape:
            y_pred     = self([noisy_inputs, sigmas], training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss       = self.compute_loss(y = y_true, y_pred = y_pred) #, sample_weight = 1./sigmas)
            # tf.print("Loss computed:", loss)
            tf.debugging.assert_all_finite(loss, "Loss contains NaN or Inf values!")
            
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y_true, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


    def test_step(self, batch):
        spectra, labels, sigmas = batch
        y_pred = self([spectra, sigmas], training=False)
        y_true = tf.concat([labels, tf.dtypes.cast(tf.reshape(sigmas, (tf.shape(labels)[-2],1)), tf.float32)], axis=-1)
        loss = self.compute_loss(y = y_true, y_pred = y_pred) #, sample_weight = 1./sigmas)
        self.compiled_metrics.update_state(y_true, y_pred)
        # Return a dict mapping metric names to current value.
        return {m.name: m.result() for m in self.metrics}