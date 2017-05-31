import tensorflow as tf
import numpy as np


#don't use first and last dimension as a data-vector

def generator_model_fn(features,labels,mode):
    features = tf.reshape(features,[-1,22])
    pass

def get_Discriminator():
    pass