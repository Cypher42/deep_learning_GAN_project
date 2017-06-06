import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import csv
from matplotlib import pyplot as plt
import progressbar
from DataProcessorr import decode

def preprocess_data(mbatch_size = 128):
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
            yield random.choice(data,mbatch_size)



   # return np.array(result)

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


X = tf.placeholder(tf.float32, shape=[None, 134])

D_W1 = tf.Variable(xavier_init([134, 66]))
D_b1 = tf.Variable(tf.zeros(shape=[66]))

D_W2 = tf.Variable(xavier_init([66, 33]))
D_b2 = tf.Variable(tf.zeros(shape=[33]))

D_W3 = tf.Variable(xavier_init([33, 1]))
D_b3 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2,D_W3,D_b1, D_b2,D_b3]


Z = tf.placeholder(tf.float32, shape=[None, 11])

G_W0 = tf.Variable(xavier_init([11, 33]))
G_b0 = tf.Variable(tf.zeros(shape=[33]))

G_W1 = tf.Variable(xavier_init([33, 66]))
G_b1 = tf.Variable(tf.zeros(shape=[66]))

G_W2 = tf.Variable(xavier_init([66, 134]))
G_b2 = tf.Variable(tf.zeros(shape=[134]))

theta_G = [G_W0,G_W1, G_W2, G_b0,G_b1, G_b2]


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    G_h0 = tf.nn.relu(tf.matmul(z,G_W0)+ G_b0)
    G_h1 = tf.nn.relu(tf.matmul(G_h0, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_logit2 = tf.matmul(D_logit, D_W3)+ D_b3
    D_prob = tf.nn.sigmoid(D_logit2)

    return D_prob, D_logit


G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))

# Alternative losses:
# -------------------
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

mb_size = 128
Z_dim = 11

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

d_l = list()
g_l = list()

bar = progressbar.ProgressBar()

for it in bar(range(100000)):
    X_mb = preprocess_data(128).__next__()
    if it%10 ==0:
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})

    if it % 10000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()
    d_l.append(D_loss_curr)
    g_l.append(G_loss_curr)

decode()

samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})
with open('results.csv','w+',newline='') as fp:
    wrtr = csv.writer(fp,delimiter=',' )
    for line in samples:
        wrtr.writerow(line)

l1 = plt.plot(d_l,label='Discriminator')
l2 = plt.plot(g_l,label='Generator')
plt.legend()
plt.show()
