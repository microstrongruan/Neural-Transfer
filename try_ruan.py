#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 12:45:45 2017

@author: ruanjiaqiang
"""
import tensorflow as tf

z = tf.random_normal((1,2,3,4))
y = tf.placeholder(tf.float32, shape=(None,))
x = tf.nn.l2_loss(y)

print(tf.Session().run(z))

#with tf.Session():
#    print(y.eval())