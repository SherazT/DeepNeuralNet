import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

no_nodes_hlyr1 = 500
no_nodes_hlyr2 = 500
no_nodes_hlyr3 = 500

n_classes = 10 #outputs
batch_size = 100

#height x width
x = tf.placeholder('float', [None,784]) #28x28 = 784 pixels (squashed)
y = tf.placeholder('float')

def neural_network_model(data):
	hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([784,  no_nodes_hlyr1])),
					  'biases': tf.Variable(tf.random_normal(no_nodes_hlyr1))}

