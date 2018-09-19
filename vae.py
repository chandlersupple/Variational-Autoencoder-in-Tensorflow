# Chandler Supple, 9/18/18

import tensorflow as tf
import numpy as np
from keras.preprocessing import image
sess = tf.InteractiveSession()

def lrelu(x, alpha= 0.4):
    return tf.maximum(x, tf.multiply(x, alpha))
lr = lrelu

class Autoencoder():
    
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 4096])
        x_4D = tf.reshape(self.x, [-1, 64, 64, 1])
        
        conv1 = tf.layers.conv2d(inputs= x_4D, filters= 64, kernel_size= (5, 5), strides= (3, 3), padding= 'SAME',
                                 activation= lr)
        conv2 = tf.layers.conv2d(inputs= conv1, filters= 64, kernel_size= (5, 5), strides= (3, 3), padding= 'SAME',
                                 activation= lr)
        conv3 = tf.layers.conv2d(inputs= conv2, filters= 64, kernel_size= (5, 5), strides= (3, 3), padding= 'SAME',
                                 activation= lr)
        flat_convE = tf.contrib.layers.flatten(conv3)
        self.mn = tf.layers.dense(flat_convE, 64, activation= lr)
        self.sd = 0.5 * tf.layers.dense(flat_convE, 64, activation= lr)
        epsilon = tf.random_normal(tf.stack([tf.shape(flat_convE)[0], 64]))
        self.E = self.mn + tf.multiply(epsilon, tf.exp(self.sd))
        
        resh_E = tf.reshape(self.E, [-1, 8, 8, 1])
        us1 = tf.image.resize_images(resh_E, size= (8, 8), method= tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv4 = tf.layers.conv2d(inputs= us1, filters= 64, kernel_size= (5, 5), padding= 'SAME',
                                 activation= lr)
        us2 = tf.image.resize_images(conv4, size= (22, 22), method= tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv5 = tf.layers.conv2d(inputs= us2, filters= 64, kernel_size= (5, 5), padding= 'SAME',
                                 activation= lr)
        us3 = tf.image.resize_images(conv5, size= (64, 64), method= tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        self.D = tf.layers.conv2d(inputs= us3, filters= 1, kernel_size= (5, 5), padding= 'SAME',
                                 activation= lr)
        
        resh_D = tf.reshape(self.D, [-1, 4096])
        self.pix_loss = tf.reduce_sum(tf.squared_difference(resh_D, self.x), 1)
        self.latent_loss = -0.5 * tf.reduce_sum(1 + 2 * self.sd - tf.square(self.mn) - tf.exp(2 * self.sd), 1)
        self.loss = tf.reduce_mean(self.pix_loss + self.latent_loss)
        
        max_norm = 10
        optimizer = tf.train.AdamOptimizer(0.0001)
        gvs = optimizer.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -1 * max_norm, max_norm), var) for grad, var in gvs]
        self.train_op = optimizer.apply_gradients(capped_gvs)
        
        sess.run(tf.global_variables_initializer())
        
    def encoder(self, x):
        return sess.run(self.E, feed_dict= {self.x: x})
        
    def decode(self, E):
        D = sess.run(self.D, feed_dict= {self.E: E})
        D_flat = np.reshape(D, [-1, 4096])
        return [int(term) for term in D_flat[0]]
        
    def batch(self, batch_size, batch_iter): # Returns a batch of images from the given dataset
        batch_iarr = []
        for inst in range (batch_iter * batch_size + 1, batch_iter * batch_size + batch_size + 1):
            image_inst = image.load_img(('Img-%s.jpg' %(inst)), color_mode= 'grayscale', target_size = [64, 64])
            # In the above line of code, change 'Img-%s.jpg' in accordance to the base-name of your images
            inst_arr = image.img_to_array(image_inst)
            inst_flat = np.reshape(inst_arr, [4096])
            batch_iarr.append(inst_flat)
        return batch_iarr

batch_size = 16
batches_in_epoch = 13000 // batch_size # 1300 corresponds to the number of images in my dataset -- modify accordingly
ae = Autoencoder() # Creates an instance of the Autoencoder() class

# Training
for epoch in range (128):
    for batch_iter in range (batches_in_epoch):
        batch_xy = ae.batch(batch_size, batch_iter)
        sess.run(ae.train_op, feed_dict= {ae.x: batch_xy})
        if batch_iter % 25 == 0:
            error = sess.run(ae.loss, feed_dict= {ae.x: batch_xy})
            print('Epoch: %s, Batch %s/%s, Loss: %s' %(epoch, batch_iter, batches_in_epoch, error))
        
def sample_compar(batch_size):
    batch_xy = ae.batch(batch_size, np.random.randint(0, 1300 - batch_size))
    E = ae.encoder(batch_xy)
    samples = []
    
    for sample_iter in range (batch_size):
        D = ae.decode([E[sample_iter]])
        D_img = np.reshape(D, [64, 64, 1])
        sample = image.array_to_img(D_img)
        sample_x = image.array_to_img(np.reshape(batch_xy[sample_iter], [64, 64, 1]))
        samples.append((sample, sample_x))
        
    return samples

# Returns a list of images in the format [(predicted_image, actual_img), ...]
samples = sample_compar(5)
