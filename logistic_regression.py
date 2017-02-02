from sklearn.linear_model import LogisticRegression
import cPickle as pickle
import numpy as np
size=50
image_size=28

print('loading data')

with open('notMNIST.pickle','rb') as f:
	data=pickle.load(f)

print('Finished Loading')
train_dt=data['train_dataset']
length=train_dt.shape[0]
train_dt=train_dt.reshape(length,image_size*image_size)
train_lb=data['train_labels']

test_dt=data['test_dataset']
length=test_dt.shape[0]
test_dt=test_dt.reshape(length,image_size*image_size)
test_lb=data['test_labels']

def train_linear_logistic(tdata,tlabel):

	model=LogisticRegression(C=1.0,penalty='l1')

	print ('initializing model size is = {}'.format(size))
	model.fit(tdata[:size,:],tlabel[:size])

	print('testing model')
	y_out=model.predict(test_dt)

	print('The accuracy of the model of size = {} is {}'.format(size,np.sum(y_out==test_lb)*1.0/len(y_out)))

	return None

train_linear_logistic(train_dt,train_lb)
size=100
train_linear_logistic(train_dt,train_lb)
size=1000
train_linear_logistic(train_dt,train_lb)
size=5000
train_linear_logistic(train_dt,train_lb)
