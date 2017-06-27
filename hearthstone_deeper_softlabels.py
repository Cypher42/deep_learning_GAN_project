import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from DataProcessor import decode
import os
import csv
import progressbar

from DataProcessor import decode_line_light

mb_size = 256
Z_dim = 10

global train
global sess
sess = None

train = False

leak = 0.2
#activation_function = tf.nn.relu

#activation_function = lambda x: tf.maximum(leak*x, x)
def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)

def selu(x,leak=0.2,name='selu'):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))

activation_function = selu

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
            yield random.sample(data,mbatch_size)

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def soft_labels(size, real):
    if real:
        return tf.random_uniform(size, minval = 0.8, maxval = 1, dtype = tf.float32)
    else:
        return tf.random_uniform(size, minval = 0, maxval = 0.2, dtype = tf.float32)


X = tf.placeholder(tf.float32, shape=[None, 134])

Z = tf.placeholder(tf.float32, shape=[None, Z_dim])


def sample_Z(m, n):
    return np.random.normal(size = (m, n))
    #return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    global sess
    layer = tf.layers.dense(z, 11, activation = activation_function)
    layer = tf.layers.dropout(layer,rate=0.4,training=train)
    #layer = tf.layers.batch_normalization(layer)
    layer = tf.layers.dense(layer, 33, activation = tf.nn.relu)
    layer = tf.layers.dropout(layer,rate=0.4,training=train)
    #layer = tf.layers.batch_normalization(layer)
    layer = tf.layers.dense(layer, 268, activation = activation_function)
    layer = tf.layers.dropout(layer,rate=0.4,training=train)
    #layer = tf.layers.batch_normalization(layer)
    #layer = tf.layers.dense(layer, 132, activation = tf.nn.relu)
    #layer = tf.layers.dropout(layer,rate=0.4,training=train)
    #layer = tf.layers.batch_normalization(layer)
    G_prob = tf.layers.dense(layer, 134, activation = tf.nn.softmax)

    # if sess != None:
    #     G_prob_arr = G_prob.eval(session=sess)
    #     for i in range(len(G_prob)):
    #         G_prob_arr[i] = decode_line_light(G_prob[i])
    #     G_prob = tf.Constant(G_prob_arr)
    return G_prob

def discriminator(x):

    x = tf.layers.batch_normalization(x)
    layer = tf.layers.dense(x, 134, activation = tf.nn.relu)
    layer = tf.layers.dropout(layer,rate=0.4,training=train)
    layer = tf.layers.batch_normalization(layer)
    layer = tf.layers.dense(x, 134, activation = activation_function)
    layer = tf.layers.dropout(layer,rate=0.4,training=train)
    layer = tf.layers.batch_normalization(layer)
    layer = tf.layers.dense(layer, 67, activation = activation_function)
    layer = tf.layers.dropout(layer,rate=0.4,training=train)
    layer = tf.layers.batch_normalization(layer)
    layer = tf.layers.dense(layer, 33, activation = activation_function)
    layer = tf.layers.dropout(layer,rate=0.4,training=train)
    layer = tf.layers.batch_normalization(layer)
    layer = tf.layers.dense(layer, 16, activation = activation_function)
    layer = tf.layers.dropout(layer,rate=0.4,training=train)
    layer = tf.layers.batch_normalization(layer)
    D_logit = tf.layers.dense(layer, 1, activation = None)

    #D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    #D_logit = tf.matmul(layer, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

""" FÜR GLEICHEN DISCRIMINATOR
G_sample = generator(Z)
discriminator_temp = tf.make_template("discriminator", discriminator)
D_real, D_logit_real = discriminator_temp(X)
D_fake, D_logit_fake = discriminator_temp(G_sample)"""
# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))

# Alternative losses:
# -------------------

"""
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels= soft_labels(tf.shape(D_logit_real), True))) #tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels= soft_labels(tf.shape(D_logit_fake), False))) #tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=soft_labels(tf.shape(D_logit_fake), True)))# tf.ones_like(D_logit_fake)))
"""
D_loss = 0.5 * (tf.reduce_mean((D_real - 1)**2) + tf.reduce_mean(D_fake**2))
G_loss = 0.5 * tf.reduce_mean((D_fake - 1)**2)

D_solver = tf.train.GradientDescentOptimizer(0.001).minimize(D_loss)#, var_list=theta_D)
G_solver = tf.train.AdamOptimizer(0.001).minimize(G_loss)#, var_list=theta_G)

"""
D_loss_real = tf.log(D_real)
D_loss_fake = tf.log(tf.subtract(tf.ones_like(D_fake), D_fake))
D_loss = - (tf.reduce_mean(tf.add(D_loss_real, D_loss_fake)))
G_loss = - tf.reduce_mean(tf.log(D_fake))
D_solver = tf.train.GradientDescentOptimizer(0.3).minimize(D_loss)#, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss)#, var_list=theta_G)
"""

"""if sess != None:
    print("d_real. ", D_loss_real.eval(session=sess))
    print("d_fake. ", D_loss_fake.eval(session=sess))
    print("g: ", G_loss.eval(session=sess))"""


sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

d_l = list()
g_l = list()
G_loss_curr = 0.0
D_loss_curr = 0.0
bar = progressbar.ProgressBar()
gen_t = 0
dis_t = 0
for it in bar(range(100000)):
    train = True
    X_mb = preprocess_data_shuffle(mb_size).__next__()
    #if it%5==0: improves the discriminator a lot
    #z = sample_Z(mb_size, Z_dim) # using same z for both doesn't change anything
    #if it%10
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
    dis_t += 1
    if D_loss_curr <= 0.15:
        gen_t += 1
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})

    #if it % 10000 == 0:
     #   print('Iter: {}'.format(it))
      #  print('D loss: {:.4}'. format(D_loss_curr))
       # print('G_loss: {:.4}'.format(G_loss_curr))
        #print()
    g_l.append(G_loss_curr)
    d_l.append(D_loss_curr)

print('Iter: {}'.format(it))
print('D loss: {:.4}'. format(D_loss_curr))
print('G_loss: {:.4}'.format(G_loss_curr))
print('Trained discriminator:',dis_t)
print("Trained generator:",gen_t)
print()


train = False

#print(d_l)

samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})
with open('results.csv','w+',newline='') as fp:
    wrtr = csv.writer(fp,delimiter=',' )
    for line in samples:
        wrtr.writerow(line)

decode()


l1 = plt.plot(d_l,label='Discriminator')
l2 = plt.plot(g_l,label='Generator')
plt.legend()
plt.show()
