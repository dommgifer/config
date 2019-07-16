import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

device_name = "/device:XLA_GPU:0"  # Choose device from cmd line. Options: gpu or cpu
shape = (1500,1500)
with tf.device(device_name):
    random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
    sum_operation = tf.reduce_sum(dot_operation)


startTime = datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
    result = session.run(sum_operation)

print(result)
