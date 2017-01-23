import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

no_nodes_hlyr1 = 500
no_nodes_hlyr2 = 500
no_nodes_hlyr3 = 500

n_classes = 10 #outputs
n_inputs = 784
batch_size = 100

#height x width
x = tf.placeholder('float', [None,n_inputs]) #28x28 = 784 pixels (squashed)
y = tf.placeholder('float')

def neural_network_model(data):
	# (input_data * weights) + biases
	hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([n_inputs,  no_nodes_hlyr1])),
					  'biases': tf.Variable(tf.random_normal(no_nodes_hlyr1))}

	hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([no_nodes_hlyr1,  no_nodes_hlyr2])),
					  'biases': tf.Variable(tf.random_normal(no_nodes_hlyr2))}

	hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([no_nodes_hlyr2,  no_nodes_hlyr3])),
					  'biases': tf.Variable(tf.random_normal(no_nodes_hlyr3))}

	output_layer = {'weights': tf.Variable(tf.random_normal([no_nodes_hlyr3, n_classes])),
					  'biases': tf.Variable(tf.random_normal(n_classes))}

	layer1 = tf.add(tf.matmul(data, hidden_layer_1['weights']) + hidden_layer_1['biases'])
	layer1 = tf.nn.relu(layer1) 	#relu is activation function

	layer2 = tf.add(tf.matmul(layer1, hidden_layer_2['weights']) + hidden_layer_2['biases'])
	layer2 = tf.nn.relu(layer2)

	layer3 = tf.add(tf.matmul(layer2, hidden_layer_3['weights']) + hidden_layer_3['biases'])
	layer3 = tf.nn.relu(layer3)

	output = tf.matmul(layer3, output_layer['weights']) + output_layer['biases']

	return output

def train_neural_network(x): #x is input
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))

	optimizer = tf.train.AdamOptimizer().minimize(cost) #default learning rate is 0.001