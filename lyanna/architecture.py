#!/usr/bin/env python

from lyanna.sansa import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

def create_Sansa(
    conv_dropout  = None, 
    l2_conv = None, 
    l2_dense = None, 
    return_pretrained = False,
):
    """
    Architecture of the Sansa model of Nayak et al. (2024).

    Args:
        conv_dropout (float, optional): dropout rate for the residual (convolutional) blocks, in keras terms, applied after each residual block and also to the activated convolutional layer of each residual block. Only important during network training. Defaults to None.
        l2_conv (float, optional): L2 regularization factor for the convolutional kernels. Only important during network training. Defaults to None.
        l2_dense (float, optional): L2 regularization factor for the final layer layer weights. Only important during network training. Defaults to None.
        return_pretrained (bool, optional): whether to return a network architecture with the first two residual blocks only, if you're interested in transfer learning from a previously trained network. Defaults to False.

    Returns:
        Sansa (tf.keras.Model): architecture of Sansa
        [pretrained part (tf.keras.Model): pretrained part of the network, if return_pretrained is True]
    """
    conv_initializer   = tf.keras.initializers.GlorotNormal(seed = 0)
    dense_initializer  = tf.keras.initializers.HeNormal(seed = 0)
    conv_regularizer   = tf.keras.regularizers.L2(l2 = l2_conv)
    dense_regularizer  = tf.keras.regularizers.L2(l2 = l2_dense)
    tf.random.set_seed(100) #*(Run_id+1))
    # model = tf.keras.Sequential()
    
    inputs = tf.keras.layers.Input(shape = (512,1))
    
    # First Residual Block
    x = ResBlock(
        16, 16, 
        strides = 2, 
        activation = 'leaky_relu', 
        padding = 'same', 
        regularizer = conv_regularizer, 
        initializer = conv_initializer, 
        dropout = conv_dropout, 
        idx = 0
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name = 'batch_norm_0')(x)
    x = tf.keras.layers.Dropout(conv_dropout, name = 'dropout_conv_0')(x)
    x = tf.keras.layers.AveragePooling1D(4, padding = 'same', name = 'avgpool_0')(x)
    
    # Second Residual Block
    x = ResBlock(
        32, 8, 
        strides = 2, 
        activation = 'leaky_relu', 
        padding = 'same', 
        regularizer = conv_regularizer, 
        initializer = conv_initializer, 
        dropout = conv_dropout, 
        idx = 1
    )(x)
    x = tf.keras.layers.BatchNormalization(name = 'batch_norm_1')(x)
    x = tf.keras.layers.Dropout(conv_dropout, name = 'dropout_conv_1')(x)
    x = tf.keras.layers.AveragePooling1D(4, padding = 'same', name = 'avgpool_1')(x)
    
    if return_pretrained:
        pretrained_model = Sansa(
            inputs = inputs, 
            outputs = x, 
            name = 'Sansa_pretrained'
        )

    # Third Residual Block
    x = ResBlock(
        32, 8, 
        strides = 1, 
        activation = 'leaky_relu', 
        padding = 'same', 
        regularizer = conv_regularizer, 
        initializer = conv_initializer, 
        dropout = conv_dropout, 
        idx = 2
    )(x)
    x = tf.keras.layers.BatchNormalization(name = 'batch_norm_2')(x)
    x = tf.keras.layers.Dropout(conv_dropout, name = 'dropout_conv_2')(x)
    x = tf.keras.layers.AveragePooling1D(2, padding = 'same', name = 'avgpool_2')(x)

    # Fourth Residual Block
    x = ResBlock(
        64, 8, 
        strides = 1, 
        activation = 'leaky_relu', 
        padding = 'same', 
        regularizer = conv_regularizer, 
        initializer = conv_initializer, 
        dropout = conv_dropout, 
        idx = 3
    )(x)
    x = tf.keras.layers.BatchNormalization(name = 'batch_norm_3')(x)
    x = tf.keras.layers.Dropout(conv_dropout, name = 'dropout_conv_3')(x)
    x = tf.keras.layers.AveragePooling1D(2, padding = 'same', name = 'avgpool_3')(x)
    
    x = tf.keras.layers.Flatten(name = 'flatten')(x)
    
    output = tf.keras.layers.Dense(
        5, 
        activation = None, 
        use_bias = False, 
        kernel_regularizer = dense_regularizer, 
        kernel_initializer = dense_initializer, 
        name = 'dense_final'
    )(x)
    
    model = Sansa(
        inputs = inputs, 
        outputs = output, 
        name = 'Sansa'
    )
    
    if return_pretrained:
        return model, pretrained_model
    else:
        return model
    


def create_nSansa(
    query = True,
    N_dense_nodes = 85, 
    conv_dropout = None, 
    dense_dropout = None, 
    l2_conv = None, 
    l2_dense = None, 
):
    """
    Architecture of the nSansa model of Nayak et al. (2025).

    Args:
        query (bool, optional): whether the architecture includes the noise sigma query. Defaults to True.
        N_dense_nodes (int, optional): number of dense nodes in the only nonlinear layer after the residual part. Defaults to 80.
        conv_dropout (float, optional): dropout rate for the residual (convolutional) blocks, in keras terms, applied after each residual block and also to the activated convolutional layer of each residual block. Only important during network training. Defaults to None.
        dense_dropout (float, optional): dropout rate for the only nonlinear layer after the residual part, in keras terms. Only important during network training. Defaults to None.
        l2_conv (float, optional): L2 regularization factor for the convolutional kernels. Only important during network training. Defaults to None.
        l2_dense (float, optional): L2 regularization factor for the dense layer weights. Only important during network training. Defaults to None.

    Returns:
        tf.keras.Model: the keras model nSansa
    """
    conv_initializer   = tf.keras.initializers.GlorotNormal(seed = 0)
    dense_initializer  = tf.keras.initializers.HeNormal(seed = 0)
    conv_regularizer   = tf.keras.regularizers.L2(l2 = l2_conv)
    dense_regularizer  = tf.keras.regularizers.L2(l2 = l2_dense)
    tf.random.set_seed(15)
    # model = tf.keras.Sequential()
    activation = 'leaky_relu'

    input_spec = tf.keras.layers.Input(shape = (256,1), name = 'input_spec')
    if query:
        input_sig  = tf.keras.layers.Input(shape = (1,), name = 'input_sig')
        a = -7.5; b = -2.5
        sigma_rescaled = (2.*tf.math.log(input_sig) - (a+b))/(b-a)
    
    # First Residual Block
    x = ResBlock(
        16, 16, 
        strides = 2, 
        activation = activation, 
        padding = 'same', 
        regularizer = conv_regularizer, 
        initializer = conv_initializer, 
        dropout = conv_dropout, 
        idx = 0
    )(input_spec)
    x = tf.keras.layers.BatchNormalization(name = 'batch_norm_0')(x)
    x = tf.keras.layers.Dropout(conv_dropout, name = 'dropout_conv_0')(x)
    x = tf.keras.layers.AveragePooling1D(4, padding = 'same', name = 'avgpool_0')(x)

    # Second Residual Block
    x = ResBlock(
        32, 8, 
        strides = 2, 
        activation = activation, 
        padding = 'same', 
        regularizer = conv_regularizer, 
        initializer = conv_initializer, 
        dropout = conv_dropout, 
        idx = 1
    )(x)
    x = tf.keras.layers.BatchNormalization(name = 'batch_norm_1')(x)
    x = tf.keras.layers.Dropout(conv_dropout, name = 'dropout_conv_1')(x)
    x = tf.keras.layers.AveragePooling1D(2, padding = 'same', name = 'avgpool_1')(x)

    # pretrained_model = nSansa(inputs = inputs, outputs = x, name = 'nSansa_pretrained')

    # Third Residual Block
    x = ResBlock(
        32, 8, 
        strides = 1, 
        activation = activation, 
        padding = 'same', 
        regularizer = conv_regularizer, 
        initializer = conv_initializer, 
        dropout = conv_dropout, 
        idx = 2
    )(x)
    x = tf.keras.layers.BatchNormalization(name = 'batch_norm_2')(x)
    x = tf.keras.layers.Dropout(conv_dropout, name = 'dropout_conv_2')(x)
    x = tf.keras.layers.AveragePooling1D(2, padding = 'same', name = 'avgpool_2')(x)

    # Fourth Residual Block
    x = ResBlock(
        64, 8, 
        strides = 1, 
        activation = activation, 
        padding = 'same', 
        regularizer = conv_regularizer, 
        initializer = conv_initializer, 
        dropout = conv_dropout, 
        idx = 3
    )(x)
    x = tf.keras.layers.BatchNormalization(name = 'batch_norm_3')(x)
    x = tf.keras.layers.Dropout(conv_dropout, name = 'dropout_conv_3')(x)
    # x = tf.keras.layers.AveragePooling1D(2, padding = 'same', name = 'avgpool_3')(x)

    # Fifth Residual Block
    x = ResBlock(
        64, 8, 
        strides = 1, 
        activation = activation, 
        padding = 'same', 
        regularizer = conv_regularizer, 
        initializer = conv_initializer, 
        dropout = conv_dropout, 
        idx = 4
    )(x)
    x = tf.keras.layers.BatchNormalization(name = 'batch_norm_4')(x)
    x = tf.keras.layers.Dropout(conv_dropout, name = 'dropout_conv_4')(x)
    # x = tf.keras.layers.AveragePooling1D(2, padding = 'same', name = 'avgpool_4')(x)
    
    # Fifth Residual Block
    x = ResBlock(
        64, 8, 
        strides = 1, 
        activation = activation, 
        padding = 'same', 
        regularizer = conv_regularizer, 
        initializer = conv_initializer, 
        dropout = conv_dropout, 
        idx = 5
    )(x)
    x = tf.keras.layers.BatchNormalization(name = 'batch_norm_5')(x)
    # x = tf.keras.layers.Dropout(conv_dropout, name = 'dropout_conv_5')(x)
    x = tf.keras.layers.AveragePooling1D(2, padding = 'same', name = 'avgpool_5')(x)

    # Flatten and include query if needed
    x = tf.keras.layers.Flatten(name = 'flatten')(x)
    if query:
        x = tf.keras.layers.Concatenate(name = 'incl_query')([x, sigma_rescaled])

    x = tf.keras.layers.Dense(
        N_dense_nodes, 
        activation = 'sigmoid', 
        use_bias = False, 
        kernel_regularizer = dense_regularizer, 
        kernel_initializer = dense_initializer, 
        name = 'dense_0'
    )(x)
    x = tf.keras.layers.BatchNormalization(name = 'batch_norm_dense_0')(x)
    x = tf.keras.layers.Dropout(dense_dropout, name = 'dropout_dense_0')(x)
    # x = tf.keras.layers.Dense(32, activation = 'sigmoid', kernel_regularizer = dense_regularizer, kernel_initializer = dense_initializer, bias_initializer = dense_initializer, name = 'dense_2')(x)
    # x = tf.keras.layers.Dropout(dense_dropout_rsansa, name = 'dropout_dense_2')(x)
    output = tf.keras.layers.Dense(
        5, 
        activation = None, 
        use_bias = False, 
        kernel_regularizer = dense_regularizer, 
        kernel_initializer = dense_initializer, 
        name = 'dense_final'
    )(x)
    if query:
        model = nSansaQuery(
            inputs = [input_spec, input_sig], 
            outputs = output, 
            name = 'nSansa',
        )
    else:
        model = nSansa(
            inputs = input_spec, 
            outputs = output, 
            name = 'nSansa',
        )
    return model #, pretrained_model #, summary