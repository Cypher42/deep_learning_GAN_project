import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from DataProcessorr import decode
import os
import csv
import progressbar
from tensorflow.contrib.framework import get_variables

from DataProcessorr import decode_line_light

sess = tf.InteractiveSession()

# PARAMETERS:
mb_size = 128
Z_dim = 10

# For Dropout:
global train
train = False

# activation functions: Either relu or leaky relu
relu = tf.nn.relu
leak = 0.2
leaky_relu = lambda x: tf.maximum(leak*x, x)
activation_function = leaky_relu

def preprocess_data(mbatch_size = 128):
    """
    Read data out of gan_card file, in which the cards are
    already formatted as one hot vectors.
    Takes the first mbatch_size lines
    """
    result = list()
    while(True):
        with open('gan_card.csv','r') as fp:
            rdr = csv.reader(fp,delimiter=',')
            c = 0
            first = True
            for line in rdr:
                if first:
                    first = False
                    continue
                c += 1
                if c > mbatch_size:
                    c = 0
                    yield np.array(result)
                    result = list()
                line = (line[1:-1])
                for i in range(len(line)):
                    line[i] = float(line[i])
                result.append(line)


def preprocess_data_shuffle(mbatch_size=128):
    """
    Read data out of gan_card file, in which the cards are
    already formatted as one hot vectors.
    Reads line by line and yields a random minibatch from these
    """
    with open('gan_card.csv','r') as fp:
        rdr = csv.reader(fp,delimiter=',')
        c = 0
        data = list()
        first = True
        for line in rdr:
            if first:
                first = False
                continue
            line = (line[1:-1])
            for i in range(len(line)):
                line[i] = float(line[i])
            data.append(line)
        import random
        while(True):
            yield random.sample(data,mbatch_size)

def soft_labels(size, real):
    """
    To create soft labels.
    If real=true: yields uniformly distributed labels between 0.8 and 1
    if real=false: yields uniformly distributed labels between 0 and 0.2
    """
    if real:
        return tf.random_uniform(size, minval = 0.8, maxval = 1.0, dtype = tf.float32)
    else:
        return tf.random_uniform(size, minval = 0, maxval = 0.2, dtype = tf.float32)

def sample_Z(m, n):
    """
    Sample random values from a standrdized normal distribution
    Get tensor of size (m,n)
    Used as input to the generator
    """
    return np.random.normal(size = (m, n))
    # return np.random.uniform(-1., 1., size=[m, n])


# Initialize the Placeholders for the real data and z space:

X = tf.placeholder(tf.float32, shape=[None, 134])

Z = tf.placeholder(tf.float32, shape=[None, Z_dim])


def generator(z):
    """
    Three layer network to generate a hearthstone card from the random
    z-tensor
    commented: possible extension of the network by two more layers
    Returns: Variable scope g_theta and generated tensor G_prob
    """
    with tf.variable_scope("generator") as scope:
        layer = tf.layers.dense(z, 11, activation = activation_function)
        tf.summary.histogram("g_1_layer", layer)
        layer = tf.layers.dropout(layer,rate=0.4,training=train)
        layer = tf.layers.batch_normalization(layer)
        #layer = tf.layers.dense(layer, 33, activation = tf.nn.relu)
        #layer = tf.layers.dropout(layer,rate=0.4,training=train)
        #layer = tf.layers.batch_normalization(layer)
        layer = tf.layers.dense(layer, 268, activation = activation_function)
        tf.summary.histogram("g_2_layer", layer)
        layer = tf.layers.dropout(layer,rate=0.4,training=train)
        layer = tf.layers.batch_normalization(layer)
        #layer = tf.layers.dense(layer, 132, activation = tf.nn.relu)
        #layer = tf.layers.dropout(layer,rate=0.4,training=train)
        #layer = tf.layers.batch_normalization(layer)
        G_prob = tf.layers.dense(layer, 134, activation = tf.nn.sigmoid)
        tf.summary.histogram("g_3_layer", layer)
        g_theta = get_variables(scope)

    return g_theta, G_prob

def discriminator(x, reuse):
    """
    Four-layer architecture for the discriminator
    x: either real data or generated cards
    reuse: if true, reuse the variables - need to be the same for fake and real cards
    returns the variable scope D_theta, the unactivated output D_logit and the
    sigmoid outout D_prob
    """
    with tf.variable_scope("discriminator", reuse=reuse) as scope:
        #x = tf.layers.batch_normalization(x)
        #layer = tf.layers.dense(x, 134, activation = tf.nn.relu)
        #layer = tf.layers.dropout(layer,rate=0.4,training=train)
        #layer = tf.layers.batch_normalization(layer)
        layer = tf.layers.dense(x, 256, activation = activation_function)
        tf.summary.histogram("d_1_layer", layer)
        layer = tf.layers.dropout(layer,rate=0.4,training=train)
        layer = tf.layers.batch_normalization(layer)
        layer = tf.layers.dense(layer, 128, activation = activation_function)
        tf.summary.histogram("d_2_layer", layer)
        layer = tf.layers.dropout(layer,rate=0.4,training=train)
        layer = tf.layers.batch_normalization(layer)
        layer = tf.layers.dense(layer, 16, activation = tf.nn.relu)
        tf.summary.histogram("d_3_layer", layer)
        D_logit = tf.layers.dense(layer, 1, activation = None)
        D_prob = tf.nn.sigmoid(D_logit)
        D_theta = get_variables(scope)

    return D_theta, D_prob, D_logit


G_theta, G_sample = generator(Z)
D_theta, D_real, D_logit_real = discriminator(X, reuse=None)
D_theta, D_fake, D_logit_fake = discriminator(G_sample, reuse=True)

# ALTERNATIVE POSSIBILITY: TEMPLATE FOR DISCRIMINATOR
# G_theta, G_sample = generator(Z)
# discriminator_temp = tf.make_template("discriminator", discriminator)
# D_theta, D_real, D_logit_real = discriminator_temp(X)
# D_theta, D_fake, D_logit_fake = discriminator_temp(G_sample)

# LOSSES:
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels= soft_labels(tf.shape(D_logit_real), True))) #tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels= soft_labels(tf.shape(D_logit_fake), False))) #tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=soft_labels(tf.shape(D_logit_fake), True)))# tf.ones_like(D_logit_fake)))

# OPTIMIZER:
#G_gradients = tf.train.AdamOptimizer(0.01).compute_gradients(G_loss, var_list=G_theta)
D_solver = tf.train.GradientDescentOptimizer(0.001).minimize(D_loss, var_list=D_theta)
G_solver = tf.train.AdamOptimizer(0.001).minimize(G_loss, var_list=G_theta)

# ALTERNATIVE LOSS FUNCTION: Original log functions from the Goodfellow paper
# D_loss_real = tf.log(D_real)
# D_loss_fake = tf.log(tf.subtract(tf.ones_like(D_fake), D_fake))
# D_loss = - (tf.reduce_mean(tf.add(D_loss_real, D_loss_fake)))
# G_loss = - tf.reduce_mean(tf.log(D_fake))
# D_solver = tf.train.GradientDescentOptimizer(0.03).minimize(D_loss, var_list=D_theta)
# G_solver = tf.train.AdamOptimizer(0.05).minimize(G_loss, var_list=G_theta)


# Write loss functions in summary for tensorboard
tf.summary.scalar("D_loss_real", D_loss_real)
tf.summary.scalar("D_loss_fake", D_loss_fake)
tf.summary.scalar("D_loss", D_loss)
tf.summary.scalar("G_loss", G_loss)
# Merge all and write to file
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("./logs/nn_logs" + '/train', sess.graph)


# TRAIN MODEL:
sess.run(tf.global_variables_initializer())


if not os.path.exists('out/'):
    os.makedirs('out/')

# save losses in lists
d_l = list()
g_l = list()

D_loss_curr = 0.0
G_loss_curr= 0.0
# Display ProgressBar
bar = progressbar.ProgressBar()

for it in bar(range(2000)):
    train = True
    X_mb = preprocess_data_shuffle(mb_size).__next__()
    # Possible changes: Train D only every 5 steps, or only if G_loss reaches a certain level
    #if it%5==0:
    #if G_loss_curr < 0.5:

    summary, _, D_loss_curr = sess.run([merged, D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})

    summary, _, G_loss_curr = sess.run([merged, G_solver, G_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})

    # add summary
    train_writer.add_summary(summary, it)

    # Append losses to lists
    d_l.append(D_loss_curr)
    g_l.append(G_loss_curr)

    # for printing the loss every step:
    #if it % 10000 == 0:
     #   print('Iter: {}'.format(it))
      #  print('D loss: {:.4}'. format(D_loss_curr))
       # print('G_loss: {:.4}'.format(G_loss_curr))
        #print()


# Show progressbar and prind losses in the end
print('Iter: {}'.format(it))
print('D loss: {:.4}'. format(D_loss_curr))
print('G_loss: {:.4}'.format(G_loss_curr))
print()

train = False

# Sample some examples from the generator and save to results.csv (as vectors)
samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})
with open('results.csv','w+',newline='') as fp:
    wrtr = csv.writer(fp,delimiter=',' )
    for line in samples:
        wrtr.writerow(line)

# Decode the one hot vectors (calls decode-function in the DataProcessorr.py
# writes the decoded generated cards to results_clean.csv
decode()

# Plot the two loss functions
l1 = plt.plot(d_l,label='Discriminator')
l2 = plt.plot(g_l,label='Generator')
plt.legend()
plt.show()
