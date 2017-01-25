#Author ==> Jeevan Thomas Koshy
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

""" Simple Multiplication """

# a=tf.placeholder("float") # Create a Symbolic variable 'a'
# b=tf.placeholder("float") # Create a Symbolic variable 'b'


# y=tf.mul(a,b)

# with tf.Session() as sess:
# 	print("%f should equal 2.0" %sess.run(y,feed_dict={a:1,b:2}))
# 	print("%f should equal 9.0" %sess.run(y,feed_dict={a:3,b:3}))



"""Linear Regression"""

# trX=np.linspace(-1,1,101)
# trY=2*trX+np.random.randn(*trX.shape)*0.33 #creates a y value which is approximately linear but with some random noise

# X=tf.placeholder("float")
# Y=tf.placeholder("float")

# def model(X,w):
# 	return tf.mul(X,w) #  this is just X*w so this model is pretty simple

# w=tf.Variable(0.0,name="Weights") # create  a shared variable for the weight matrix
# y_model=model(X,w)

# cost =tf.square(Y-y_model)

# train_op=tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# with tf.Session() as sess:
# 	# In this case initialising the weights of w
# 	tf.global_variables_initializer().run()

# 	for i in range(100):
# 		for(x,y) in zip(trX,trY):
# 			sess.run(train_op,feed_dict={X:x,Y:y})

# 	print(sess.run(w))


"""logistic Regression """

# def init_weights(shape):
# 	return tf.Variable(tf.random_normal(shape,stddev=0.01))

# def model(X,w):
# 	return tf.matmul(X,w)

# mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
# trX,trY,teX,teY=mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels

# X=tf.placeholder("float",[None,784])
# Y=tf.placeholder("float",[None,10])

# w=init_weights([784,10])
# py_x=model(X,w)

# cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x,labels=Y))
# train_op=tf.train.GradientDescentOptimizer(0.05).minimize(cost)
# predict_op=tf.argmax(py_x,1)


# with tf.Session() as sess:
# 	tf.global_variables_initializer().run()

# 	for i in range(100):
# 		for start,end in zip(range(0,len(trX),128),range(128,len(trX)+1,128)):
# 			sess.run(train_op,feed_dict={X:trX[start:end],Y:trY[start:end]})
# 		print(i,np.mean(np.argmax(teY,axis=1)==sess.run(predict_op,feed_dict={X:teX})))	

"""Feed forward Neural Network"""

# def init_weights(shape):
# 	return tf.Variable(tf.random_normal(shape,stddev=0.01))

# def	model(X,w_h,w_o):
# 	h=tf.nn.sigmoid(tf.matmul(X,w_h)) 
# 	return tf.matmul(h,w_o)

# mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
# trX,trY,teX,teY=mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels

# X=tf.placeholder("float",[None,784])
# Y=tf.placeholder("float",[None,10])

# w_h=init_weights([784,625])
# w_o=init_weights([625,10])

# py_x=model(X,w_h,w_o)

# cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x,labels=Y))
# train_op=tf.train.GradientDescentOptimizer(0.05).minimize(cost)
# predict_op=tf.argmax(py_x,1)

# with tf.Session() as sess:
# 	tf.global_variables_initializer().run()
# 	for i in range(10000):
# 		for start,end in zip(range(0,len(trX),128),range(128,len(trX)+1,128)):
# 			sess.run(train_op,feed_dict={X:trX[start:end],Y:trY[start:end]})
# 		print(i,np.mean(np.argmax(teY,axis=1)==sess.run(predict_op,feed_dict={X:teX})))	

"""Deep Feed Forward neural Network"""

def init_weights(shape):
	return tf.Variable(tf.random_normal(shape,stddev=0.01))

def model(X,w_h,w_h2,w_o,p_keep_input,p_keep_hidden):
	X=tf.nn.dropout(X,p_keep_input)
	h=tf.nn.relu(tf.matmul(X,w_h))

	h=tf.nn.dropout(h,p_keep_hidden)
	h2=tf.nn.relu(tf.matmul(h,w_h2))

	h2=tf.nn.dropout(h2,p_keep_hidden)

	return tf.matmul(h2,w_o)

mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
trX,trY,teX,teY=mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels

X=tf.placeholder("float",[None,784])
Y=tf.placeholder("float",[None,10])

w_h=init_weights([784,625])
w_h2=init_weights([625,625])

w_o=init_weights([625,10])

p_keep_hidden=tf.placeholder("float")
p_keep_input=tf.placeholder("float")

py_x = model(X,w_h,w_h2,w_o,p_keep_input,p_keep_hidden)

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x,labels=Y))
train_op=tf.train.RMSPropOptimizer(0.001,0.9).minimize(cost)

predict_op=tf.argmax(py_x,1)


with tf.Session() as sess:
	for start,end in zip(range(0,len(trX),128),(128,len(trX)+1,128)):
		sess.run(train_op,feed_dict={X:trX[start:end],Y:trY[start:end],p_keep_input:0.8,p_keep_hidden:0.5})

	print(i,np.mean(np.argmax(teY,axis=1)==sess.run(predict_op,feed_dict={X:teX,p_keep_input:1.0,p_keep_hidden:1.0})))		
