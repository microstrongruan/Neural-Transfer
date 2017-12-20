# -*- coding: utf-8 -*-

from PIL import Image
from sys import stderr

import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc
from functools import reduce

CONTENT_LAYERS = ('relu4_2', 'relu5_2')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
LEARNING_RATE = 1e1
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
ITERATIONS = 1000
LAMBDA = 0.9

######################################################################

def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:,:,:3]
    return img


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)

######################################################################

VGG19_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
)

def load_net(data_path):
    data = scipy.io.loadmat(data_path)
    if not all(i in data for i in ('layers', 'classes', 'normalization')):
        raise ValueError("You're using the wrong VGG19 data. Please follow the instructions in the README to download the correct data.")
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = data['layers'][0]
    return weights, mean_pixel

def net_preloaded(weights, input_image, pooling):
    net = {}
    current = input_image
    for i, name in enumerate(VGG19_LAYERS):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current, pooling)
        net[name] = current

    assert len(net) == len(VGG19_LAYERS)
    return net

def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
            padding='SAME')
    return tf.nn.bias_add(conv, bias)


def _pool_layer(input, pooling):
    if pooling == 'avg':
        return tf.nn.avg_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                padding='SAME')
    else:
        return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                padding='SAME')

######################################################################

def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)

def _span_tensor(tensor):
    return tf.reshape(tensor, (-1, tensor.shape[-1].value))

def _span_array(array):
    return np.reshape(array, (-1, array.shape[-1]))
######################################################################

def main():
    vgg_weights, mean_pixel = load_net('imagenet-vgg-verydeep-19.mat')
    _content = imread('examples/1-content.jpg') - mean_pixel
    _shape = _content.shape
    _style = scipy.misc.imresize(imread('examples/1-style.jpg'), _shape) - mean_pixel
    shape = (1,) + _shape
            
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        content = tf.placeholder(tf.float32, shape = shape)
        content_net = net_preloaded(vgg_weights, content, 'max')
        content_feature = {}
        for layer in CONTENT_LAYERS:
            content_feature[layer] = content_net[layer].eval(feed_dict={content:np.array([_content], dtype = 'float32')})
    
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        style = tf.placeholder(tf.float32, shape = shape)
        style_net = net_preloaded(vgg_weights, style, 'max')
        style_feature = {}
        for layer in STYLE_LAYERS:
            style_feature[layer] = style_net[layer].eval(feed_dict={style:np.array([_style], dtype = 'float32')})
    
    style_gram = {}
    for layer in STYLE_LAYERS:
        temp = _span_array(style_feature[layer])
        style_gram[layer] = np.matmul(temp.T, temp)/ temp.size
    
    def print_progress():
        stderr.write('  content loss: %g\n' % content_loss.eval())
        stderr.write('    style loss: %g\n' % style_loss.eval())
        stderr.write('    total loss: %g\n' % loss.eval())
    
    with tf.Graph().as_default():
        image = tf.Variable(tf.random_normal(shape)*0.256, dtype = tf.float32)
        image_net = net_preloaded(vgg_weights, image, 'max')

        content_loss = 0
        content_losses = []
        for layer in CONTENT_LAYERS:
            content_losses.append(tf.nn.l2_loss((image_net[layer]-content_feature[layer])/content_feature[layer].size))
        content_loss += reduce(tf.add, content_losses)/len(content_losses)
    
        style_loss = 0
        style_losses = []
        for layer in STYLE_LAYERS:
            temp = _span_tensor(image_net[layer])
            image_gram = tf.matmul(tf.transpose(temp), temp) / _tensor_size(temp)
            style_losses.append(tf.nn.l2_loss((image_gram-style_gram[layer])/style_gram[layer].size))
        style_loss += reduce(tf.add, style_losses)/len(style_losses)
        
        
        Lambda = tf.Variable(0, dtype = tf.float32)
        Lambda_trans = tf.sigmoid(Lambda)
        
        loss = Lambda_trans * content_loss + (1-Lambda_trans) * style_loss
        #loss = content_loss + style_loss
        train_step = tf.train.AdamOptimizer(LEARNING_RATE,BETA1,BETA2,EPSILON).minimize(loss)
    
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(ITERATIONS):
                stderr.write('Iteration %4d/%4d\n' % (i + 1, ITERATIONS))
                train_step.run()
                if i % 50 == 0:
                    print_progress()
                if i % 50 == 0:
                    imsave('outcome/ruan_try_'+str(i)+'.jpg', np.reshape(image.eval(),shape[1:])+mean_pixel)
    print('finished')
    return 0
    
if __name__ == '__main__':
    main()