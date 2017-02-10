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

# with graph.as_default():
# 	tf_train_dataset=tf.constant(train_dataset[:train_subset,:])
# 	tf_train_labels=tf.constant(train_labels[:train_subset])
# 	tf_valid_dataset=tf.constant(train_dataset)
# 	tf_test_dataset=tf.constant(test_dataset)

# 	weights=tf.Variable(tf.truncated_normal([image_size*image_size,num_labels]))
# 	biases=tf.Variable(tf.zeros([num_labels]))

# 	logits=tf.matmul(tf_train_dataset,weights) + biases

# 	loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,logits=logits))
# 	optimiser=tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# 	train_prediction=tf.nn.softmax(logits)
# 	valid_prediction=tf.nn.softmax(tf.matmul(tf_valid_dataset,weights)+biases)

# 	test_prediction=tf.nn.softmax(tf.matmul(tf_train_dataset,weights)+biases)	


num_steps=801

def accuracy(predictions,labels):
	return (100.0*np.sum(np.argmax(predictions,1)==np.argmax(labels,1))/predictions.shape[0])

# with tf.Session(graph=graph) as session:
# 	tf.global_variable_initializer.run()
# 	print('Initilized')

# 	for step in range(num_steps):
# 		_,l,predictions=session.run([optimiser,loss,train_prediction])

# 		if (step%100==0):
# 			print("Loss at step %d : %f " %(step,l))
# 			print("Training accuracy: %.1f%%" %accuracy(predictions,train_labels[train_subset,:]))

# 			print("Validation accuracy :%.1f%%" %accuracy([valid_prediction.eval(),valid_labels]))
# 			print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


batch_size=128

graph=tf.graph

# with graph.as_default():
# 	tf_train_dataset=tf.placeholder(tf.float32,shape=(batch_size,image_size*image_size))
# 	tf_train_labels=tf.placeholder(tf.float32,shape=(batch_size,num_labels))
# 	tf_train_dataset=tf.placeholder(tf.float32,shape=(batch_size,image_size*image_size))
# 	tf_test_dataset=tf.placeholder(tf.float32,shape=(batch_size,image_size*image_size))

# 	weights=tf.Variable(tf.truncated_normal([image_size*image_size,num_labels]))
# 	biases=tf.Variable(tf.zeros([num_labels]))

# 	logits=tf.matmul(tf_train_dataset,weights) + biases

# 	loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,logits=logits))
# 	optimiser=tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# 	train_prediction=tf.nn.softmax(logits)
# 	valid_prediction=tf.nn.softmax(tf.matmul(tf_valid_dataset,weights)+biases)

# 	test_prediction=tf.nn.softmax(tf.matmul(tf_train_dataset,weights)+biases)

w_h=tf.Variable(tf.random_normal([784,1024], stddev=0.01))
w_o=tf.Variable(tf.random_normal([1024,10], stddev=0.01))

# Have to design it as a with a model in tensorflow


with graph.as_default():
	tf_train_dataset=tf.placeholder(tf.float32,shape=(batch_size,image_size*image_size))
	tf_train_labels=tf.placeholder(tf.float32,shape=(batch_size,num_labels))
	tf_train_dataset=tf.placeholder(tf.float32,shape=(batch_size,image_size*image_size))
	tf_test_dataset=tf.placeholder(tf.float32,shape=(batch_size,image_size*image_size))
	#weights=tf.Variable(tf.truncated_normal([image_size*image_size,num_labels]))
	
	hidden_layer = tf.nn.relu(tf.matmul(X, w_h))
	logits=tf.matmul(hidden_layer,w_o)
	loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,logits=logits))
	optimiser=tf.train.GradientDescentOptimizer(0.5).minimize(loss)
	train_prediction=tf.nn.softmax(logits)



num_steps=3001

with tf.Session(graph=graph) as session:
	tf.global_variable_initializer().run()
	print("Initilized")
	for step in range(num_steps):

		offset=(step*batch_size)%(train_labels.shape[0]-batch_size)
		batch_data=train_dataset[offset:(offset+batch_size),:]
		batch_labels=train_labels[offset:(batch_size+offset),:]
		feed_dict={tf_train_dataset: batch_data,tf_train_labels: batch_labels}
		_,l,predictions=session.run([optimiser,loss,train_prediction],feed_dict=feed_dict)
		if(step%500==0):
			print("Minibatch loss at step %d: %f" % (step, l))
			print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
			print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
	print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


