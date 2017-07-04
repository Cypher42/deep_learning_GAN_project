import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from DataProcessor import decode
import os
import csv
import progressbar
from tensorflow.contrib.framework import get_variables

#from DataProcessorr import decode_line_light

sess = tf.InteractiveSession()

mb_size = 128
Z_dim = 10

global train
# global sess
#sess = None

train = False

leak = 0.2
#activation_function = tf.nn.relu

activation_function = lambda x: tf.maximum(leak*x, x)

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
        return tf.random_uniform(size, minval = 0.8, maxval = 1.2, dtype = tf.float32)
    else:
        return tf.random_uniform(size, minval = 0, maxval = 0.2, dtype = tf.float32)


X = tf.placeholder(tf.float32, shape=[None, 134],name="X")

Z = tf.placeholder(tf.float32, shape=[None, Z_dim],name="Z")


def sample_Z(m, n):
    return np.random.normal(size = (m, n))
    #return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    global sess
    with tf.variable_scope("generator") as scope:
        layer = tf.layers.dense(z, 11, activation = activation_function,name='Gen_Dense_Layer_11_Units')
        tf.summary.histogram("g_1_layer", layer)
        layer = tf.layers.dropout(layer,rate=0.4,training=train,name='Gen_Dropout_Layer_1')
        layer = tf.layers.batch_normalization(layer,name='Gen_Batch_Norm_1')
        #layer = tf.layers.dense(layer, 33, activation = tf.nn.relu)
        #layer = tf.layers.dropout(layer,rate=0.4,training=train)
        #layer = tf.layers.batch_normalization(layer)
        layer = tf.layers.dense(layer, 268, activation = activation_function,name='Gen_Dense_Layer_268_Units')
        tf.summary.histogram("g_2_layer", layer)
        layer = tf.layers.dropout(layer,rate=0.4,training=train,name='Gen_Dropout_Layer_2')
        layer = tf.layers.batch_normalization(layer, name='Gen_Batch_Norm_2')
        #layer = tf.layers.dense(layer, 132, activation = tf.nn.relu)
        #layer = tf.layers.dropout(layer,rate=0.4,training=train)
        #layer = tf.layers.batch_normalization(layer)
        G_prob = tf.layers.dense(layer, 134, activation = tf.nn.sigmoid,name='Gen_Dense_134_Units')
        tf.summary.histogram("g_3_layer", layer)
        g_theta = get_variables(scope)
    # if sess != None:
    #     G_prob_arr = G_prob.eval(session=sess)
    #     for i in range(len(G_prob)):
    #         G_prob_arr[i] = decode_line_light(G_prob[i])
    #     G_prob = tf.Constant(G_prob_arr)
    return g_theta, G_prob

def discriminator(x, reuse):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse ==False
        #x = tf.layers.batch_normalization(x)
        #layer = tf.layers.dense(x, 134, activation = tf.nn.relu)
        #layer = tf.layers.dropout(layer,rate=0.4,training=train)
        #layer = tf.layers.batch_normalization(layer)
        layer = tf.layers.dense(x, 256, activation = activation_function,name='Dense_Layer_256_Units')
        tf.summary.histogram("d_1_layer", layer)
        layer = tf.layers.dropout(layer,rate=0.4,training=train,name='Dropout_Layer_1')
        layer = tf.layers.batch_normalization(layer,name='Batch_Norm_1')
        layer = tf.layers.dense(layer, 128, activation = activation_function,name='Dense_Layer_128_Units')
        tf.summary.histogram("d_2_layer", layer)
        layer = tf.layers.dropout(layer,rate=0.4,training=train,name="Droput_Layer_2")
        layer = tf.layers.batch_normalization(layer,name='Batch_Norm_2')
        layer = tf.layers.dense(layer, 16, activation = tf.nn.relu,name='Dense_Layer_16_Units')
        tf.summary.histogram("d_3_layer", layer)
        D_logit = tf.layers.dense(layer, 1, activation = None,name="logit_Layer")

        #D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
        #D_logit = tf.matmul(layer, D_W2) + D_b2
        D_prob = tf.nn.sigmoid(D_logit)
        D_theta = get_variables(scope)

    return D_theta, D_prob, D_logit


G_theta, G_sample = generator(Z)
D_theta_1, D_real, D_logit_real = discriminator(X, reuse=False)
D_theta, D_fake, D_logit_fake = discriminator(G_sample, reuse=True)
#assert(D_logit_fake == D_logit_real)

# für gleichen discriminator
"""
G_theta, G_sample = generator(Z)
discriminator_temp = tf.make_template("discriminator", discriminator)
D_theta, D_real, D_logit_real = discriminator_temp(X)
D_theta, D_fake, D_logit_fake = discriminator_temp(G_sample)
# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))
"""
# Alternative losses:
# -------------------

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels= soft_labels(tf.shape(D_logit_real), True))) #tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels= soft_labels(tf.shape(D_logit_fake), False))) #tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=soft_labels(tf.shape(D_logit_fake), True)))# tf.ones_like(D_logit_fake)))

#G_gradients = tf.train.AdamOptimizer(0.01).compute_gradients(G_loss, var_list=G_theta)
D_solver = tf.train.GradientDescentOptimizer(0.01).minimize(D_loss, var_list=D_theta)
G_solver = tf.train.AdamOptimizer(0.001).minimize(G_loss, var_list=G_theta)

"""
D_loss_real = tf.log(D_real)
D_loss_fake = tf.log(tf.subtract(tf.ones_like(D_fake), D_fake))
D_loss = - (tf.reduce_mean(tf.add(D_loss_real, D_loss_fake)))
G_loss = - tf.reduce_mean(tf.log(D_fake))
D_solver = tf.train.GradientDescentOptimizer(0.03).minimize(D_loss, var_list=D_theta)
G_solver = tf.train.AdamOptimizer(0.05).minimize(G_loss, var_list=G_theta)
"""
"""if sess != None:
    print("d_real. ", D_loss_real.eval(session=sess))
    print("d_fake. ", D_loss_fake.eval(session=sess))
    print("g: ", G_loss.eval(session=sess))"""

tf.summary.scalar("D_loss_real", D_loss_real)
tf.summary.scalar("D_loss_fake", D_loss_fake)
tf.summary.scalar("D_loss", D_loss)
tf.summary.scalar("G_loss", G_loss)

merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter("./logs/nn_logs" + '/train', sess.graph)

#sess = tf.Session()
sess.run(tf.global_variables_initializer())


#test_writer = tf.summary.FileWriter("./logs/nn_logs" + '/test')

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

d_l = list()
g_l = list()
gr = list()
D_loss_curr = 0.0
G_loss_curr= 0.0
bar = progressbar.ProgressBar()
for it in bar(range(2000)):
    train = True
    X_mb = preprocess_data_shuffle(mb_size).__next__()
    #if it%5==0: #improves the discriminator a lot
    #z = sample_Z(mb_size, Z_dim) # using same z for both doesn't change anything
    if G_loss_curr < 1.0:
        summary, _, D_loss_curr = sess.run([merged, D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})

    summary, _, G_loss_curr = sess.run([merged, G_solver, G_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})

    train_writer.add_summary(summary, it)
    #gradients = sess.run([G_gradients], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
    #gr.append(gradients)
    #if it % 10000 == 0:
     #   print('Iter: {}'.format(it))
      #  print('D loss: {:.4}'. format(D_loss_curr))
       # print('G_loss: {:.4}'.format(G_loss_curr))
        #print()
    #if it%5==0:
    d_l.append(D_loss_curr)
    g_l.append(G_loss_curr)

print('Iter: {}'.format(it))
print('D loss: {:.4}'. format(D_loss_curr))
print('G_loss: {:.4}'.format(G_loss_curr))
print()


#print("dis: ", d_l)
#print("gen: ", g_l)
#print(gr)

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
