#! /usr/bin/env python

import tensorflow as tf
from tensorflow import keras

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

class NL3_2P(keras.losses.Loss):
    """
    This class is an implementation of the negative log-likelihood loss function. The inverse covariance matrix is decomposed into Cholesky lower-triangular matrix L and its transpose. The network outputs a 5-size vecrtor y with y[0,1] are the point estimates and y[2] = L00, y[3] = L11, y[4] = L10. 
    """    
    def call(self, y_true, y_pred):
        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        delta_1 = y_true[0] - y_pred[0]
        delta_2 = y_true[1] - y_pred[1]
        return -tf.math.log(tf.math.square(y_pred[2]*y_pred[3])) + tf.math.square(y_pred[2]*delta_1) + 2.*y_pred[2]*y_pred[4]*delta_1*delta_2 + tf.math.square(y_pred[3]*delta_2) + tf.math.square(y_pred[4]*delta_2)
    
    def NL3(self, y_true, y_pred):
        return self.call(y_true, y_pred)
    
    def MeanOverallErrorT0(self, y_true, y_pred):
        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        return y_true[0] - y_pred[0]
    
    def MeanOverallErrorGamma(self, y_true, y_pred):
        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        return y_true[1] - y_pred[1]
    
    def DeterminantSigma(self, y_true, y_pred):
        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        return 1./tf.math.square(y_pred[2]*y_pred[3])
    
    def LogDeterminantSigma(self, y_true, y_pred):
        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        return -tf.math.log(tf.math.square(y_pred[2]*y_pred[3]))
    
    def MeanSquaredError(self, y_true, y_pred):
        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        return tf.math.square(y_true[0] - y_pred[0]) + tf.math.square(y_true[1] - y_pred[1])

    def ChiSquaredError(self, y_true, y_pred):
        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        delta_1 = y_true[0] - y_pred[0]
        delta_2 = y_true[1] - y_pred[1]
        return tf.math.square(tf.exp(y_pred[2])*delta_1) + 2.*tf.exp(y_pred[2])*y_pred[4]*delta_1*delta_2 + tf.math.square(tf.exp(y_pred[3])*delta_2) + tf.math.square(y_pred[4]*delta_2)
    
    def NL3andMSE(self, y_true, y_pred):
        return self.call(y_true, y_pred) + self.MeanSquaredError(y_true, y_pred)
    
    

class NL3_2P_positive(keras.losses.Loss):
    """
    This class is an implementation of the negative log-likelihood loss function. The inverse covariance matrix is decomposed into Cholesky lower-triangular matrix L and its transpose. The network outputs a 5-size vecrtor y with y[0,1] are the point estimates and y[2] = log(L00), y[3] = log(L11), y[4] = L10 such that the diagonal entries of L are always positive. 
    """
    def call(self, y_true, y_pred):
        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        delta_1 = y_true[0] - y_pred[0]
        delta_2 = y_true[1] - y_pred[1]
        return -2.*(y_pred[2] + y_pred[3]) + tf.math.square(tf.exp(y_pred[2])*delta_1) + 2.*tf.exp(y_pred[2])*y_pred[4]*delta_1*delta_2 + tf.math.square(tf.exp(y_pred[3])*delta_2) + tf.math.square(y_pred[4]*delta_2)
    
    def NL3(self, y_true, y_pred):
        return self.call(y_true, y_pred)
    
    def MeanOverallErrorT0(self, y_true, y_pred):
        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        return y_true[0] - y_pred[0]
    
    def MeanOverallErrorGamma(self, y_true, y_pred):
        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        return y_true[1] - y_pred[1]
    
    def DeterminantSigma(self, y_true, y_pred):
        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        return tf.exp(-2.*(y_pred[2] + y_pred[3]))
    
    def LogDeterminantSigma(self, y_true, y_pred):
        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        return -2.*(y_pred[2] + y_pred[3])
    
    def MeanSquaredError(self, y_true, y_pred):
        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        return tf.math.square(y_true[0] - y_pred[0]) + tf.math.square(y_true[1] - y_pred[1])

    def ChiSquaredError(self, y_true, y_pred):
        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        delta_1 = y_true[0] - y_pred[0]
        delta_2 = y_true[1] - y_pred[1]
        return tf.math.square(tf.exp(y_pred[2])*delta_1) + 2.*tf.exp(y_pred[2])*y_pred[4]*delta_1*delta_2 + tf.math.square(tf.exp(y_pred[3])*delta_2) + tf.math.square(y_pred[4]*delta_2)
    
    def NL3andMSE(self, y_true, y_pred):
        return self.call(y_true, y_pred) + self.MeanSquaredError(y_true, y_pred)
    
    def Cost(self, y_true, y_pred):
        return self.call(y_true, y_pred) + tf.abs(self.ChiSquaredError(y_true, y_pred) - 2.)
    


class NL3_2P_positive_noisy(keras.losses.Loss):
    """
    A version of the NL3_2P_positive, applicable to spectra with a known noise level (homoscedastic). The network-predicted covariance is augmented by a term proportional to the noise variance, i.e., C_total = C_predicted + l*sigma^2*I, where l is a regularization hyperparameter that is chosen to be smaller than 1 such that the second order terms in l can be neglected. The negative log likelihood is expanded to linear order in l.
    """
    def __init__(self, l = None):
        super().__init__()
        self.l = l
        print("self.l: ", self.l)

    def call(self, y_true, y_pred):
        return self.LogDeterminantSigma(y_true, y_pred) + self.ChiSquaredError(y_true, y_pred)

        # y_true = tf.transpose(y_true)
        # y_pred = tf.transpose(y_pred)
        # delta_1 = y_true[0] - y_pred[0]
        # delta_2 = y_true[1] - y_pred[1]
        # L11 = tf.exp(y_pred[2])
        # L22 = tf.exp(y_pred[3])
        # L12 = y_pred[4]
        # x = tf.math.square(L11) + tf.math.square(L22) + tf.math.square(L12)
        # P = tf.math.square(delta_1)*(tf.math.square(tf.math.square(L11)) + tf.math.square(L11) * tf.math.square(L12)) + 2.*delta_1*delta_2*L11*L12*(tf.math.square(L11) + tf.math.square(L22) + tf.math.square(L12)) + tf.math.square(delta_2)*tf.math.square( tf.math.square(L12) + tf.math.square(L22))
        # corr_term = self.l*tf.math.square(sigmas) * ( x - P ) 
        # return -2.*(y_pred[2] + y_pred[3]) + tf.math.square(tf.exp(y_pred[2])*delta_1) + 2.*tf.exp(y_pred[2])*y_pred[4]*delta_1*delta_2 + tf.math.square(tf.exp(y_pred[3])*delta_2) + tf.math.square(y_pred[4]*delta_2) + corr_term
    
    def NL3(self, y_true, y_pred):
        return self.call(y_true, y_pred)
    
    def LogDeterminantSigma(self, y_true, y_pred):
        # y_true, sigmas = y_true
        sigmas = y_true[..., -1:]  # Last column is sigmas
        y_true = y_true[..., :-1]  # All but last column is labels
        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        # print("y_pred shape:", y_pred.shape)
        sigmas = tf.transpose(sigmas)[0]
        L11 = tf.exp(y_pred[2])
        L22 = tf.exp(y_pred[3])
        L12 = y_pred[4]
        x = tf.math.square(L11) + tf.math.square(L22) + tf.math.square(L12)
        # tf.print("x:", x)
        # print(x)
        return -2.*(y_pred[2] + y_pred[3]) + self.l*tf.math.square(sigmas)*x
    
    def MeanSquaredError(self, y_true, y_pred):
        # sigmas = y_true[..., -1:]  # Last column is sigmas
        y_true = y_true[..., :-1]  # All but last column is labels
        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        return tf.math.square(y_true[0] - y_pred[0]) + tf.math.square(y_true[1] - y_pred[1])

    def ChiSquaredError(self, y_true, y_pred):
        # print("y_true shape: ", y_true.shape)
        sigmas = y_true[..., -1:]  # Last column is sigmas
        # print("sigmas shape: ", sigmas.shape)
        y_true = y_true[..., :-1]  # All but last column is labels
        # print("labels shape: ", y_true.shape)
        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        sigmas = tf.transpose(sigmas)[0]
        # print("y_true[0] shape:", y_true[0].shape)
        # print("sigmas shape:", sigmas.shape)
        delta_1 = y_true[0] - y_pred[0]
        delta_2 = y_true[1] - y_pred[1]
        L11 = tf.exp(y_pred[2])
        L22 = tf.exp(y_pred[3])
        L12 = y_pred[4]
        P = tf.math.square(delta_1)*(tf.math.square(tf.math.square(L11)) + tf.math.square(L11) * tf.math.square(L12)) + 2.*delta_1*delta_2*L11*L12*(tf.math.square(L11) + tf.math.square(L22) + tf.math.square(L12)) + tf.math.square(delta_2)*tf.math.square( tf.math.square(L12) + tf.math.square(L22))
        # tf.print("P:", P)
        return tf.math.square(tf.exp(y_pred[2])*delta_1) + 2.*tf.exp(y_pred[2])*y_pred[4]*delta_1*delta_2 + tf.math.square(tf.exp(y_pred[3])*delta_2) + tf.math.square(y_pred[4]*delta_2) - self.l*tf.math.square(sigmas)*P