from __future__ import print_function
import numpy as np
import tensorflow as tf 
from six.moves import cPickle as pickle
from six.moves import range

pickle_file= 'notMNIST.pickle'

with open(pickle_file,'rb') as f:
	save=pickle.load(f)
	train_dataset=save['train_dataset']
	train_labels=save['train_labels']
	valid_dataset=save['valid_dataset']
	valid_labels=save['valid_labels']
	test_dataset=save['test_dataset']
	test_labels=save['test_labels']

	del save

	print('Training Set ,',train_dataset.shape, train_labels.shape)
	print('validation set ',valid_dataset.shape,valid_labels.shape)
	print('Test Set ',test_dataset.shape,test_labels.shape)


image_size=28*28
num_labels=10

def reformat(dataset,labels):
	dataset=dataset.reshape(-1,image_size*image_size).astype(np.float32)
	labels=(np.arrange(num_labels) == labels[:None]).astype(np.float32)

	return dataset,labels

train_dataset,train_labels=reformat(train_dataset,train_labels)
valid_dataset,valid_labels=reformat(valid_dataset,valid_labels)
test_dataset,test_labels=reformat(test_dataset,test_labels)

print('Training Set ',train_dataset.shape,train_labels.shape)
print('Validation Set',valid_dataset.shape,valid_labels.shape)
print('Test Set',test_dataset.shape,test_labels.shape)


train_subset=10000

with graph.as_default():
	tf_train_dataset=tf.constant(train_dataset[:train_subset,:])
	tf_train_labels=tf.constant(train_labels[:train_subset])
	tf_valid_dataset=tf.constant(train_dataset)
	tf_test_dataset=tf.constant(test_dataset)

	weights=tf.Variable(tf.truncated_normal([image_size*image_size,num_labels]))
	biases=tf.Variable(tf.zeros([num_labels]))

	logits=tf.matmul(tf_train_dataset,weights) + biases

	loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,logits=logits))
	optimiser=tf.train.GradientDescentOptimizer(0.5).minimize(loss)

	train_prediction=tf.nn.softmax(logits)
	valid_prediction=tf.nn.softmax(tf.matmul(tf_valid_dataset,weights)+biases)

	test_prediction=tf.nn.softmax(tf.matmul(tf_train_dataset,weights)+biases)	


num_steps=801

def accuracy(predictions,labels):
	return (100.0*np.sum(np.argmax(predictions,)))