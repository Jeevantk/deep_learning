from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os,random
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import cv2


url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None

def download_progress_hook(count, blockSize, totalSize):
	global last_percent_reported
 	percent = int(count * blockSize * 100 / totalSize)

	if last_percent_reported != percent:
		if percent % 5 == 0:
			sys.stdout.write("%s%%" % percent)
			sys.stdout.flush()
		else:
			sys.stdout.write(".")
			sys.stdout.flush()
      
		last_percent_reported = percent

def maybe_download(filename, expected_bytes, force=False):
	if force or not os.path.exists(filename):
		print('Attempting to download:', filename) 
		filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
		print('\nDownload Complete!')
	statinfo = os.stat(filename)
	if statinfo.st_size == expected_bytes:
		print('Found and verified', filename)
	else:
		raise Exception('Failed to verify ' + filename + '.Can you get to it with a browser?')
	return filename

train_filename=maybe_download('notMNIST_large.tar.gz', 247336696)	
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)


num_classes=10
np.random.seed(133)

def maybe_extract(filename,force=False):
	root=os.path.splitext(os.path.splitext(filename)[0])[0]  # remove tar.gz
	if os.path.isdir(root) and not force:
		print('%s already present - Skipping extraction of %s' %(root,filename))
	else:
		print('Extracting data for %s. This may take a while. Please wait.'%root)
		tar=tarfile.open(filename)
		sys.stdout.flush()
		tar.extractall()
		tar.close()

	data_folders=[os.path.join(root,d) for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root,d))]

	if len(data_folders) != num_classes:
		raise Exception('Expected %d folders , one per class . FOund %d instead.' %(num_classes,len(data_folders)))

	print(data_folders)

	return data_folders

train_folders=maybe_extract(train_filename)
test_folders=maybe_extract(test_filename)

# Problem number 1 : Display some of these saved images randomly so as  to check if we  did the correct thing

dir_name="notMNIST_large"
folder_names=["A","B","C","D","E","F","G","H","I","J"]
for folder in folder_names:
	im_name=random.choice(os.listdir(dir_name+'/'+folder))
	im_file=dir_name+'/'+folder+'/'+im_name
	img=cv2.imread(im_file)
	cv2.imshow("Image",img)
	cv2.waitKey(0)

cv2.destroyAllWindows()	

image_size=28
pixel_depth=255.0

def load_letter(folder,min_num_images):
	"""Load the data for a single letter label"""
	image_files=os.listdir(folder)
	dataset=np.ndarray(shape=len(image_files),image_size,image_size,dtype=np.float32)
	print(folder)
	num_images=0
	for image in image_files:
		image_file=os.path.join(folder,image)
		try:
			image_data=(ndimage.imread(image_file).astype(float)-pixel_depth/2)/pixel_depth
			if image_data.shape !=(image_size,image_size):
				raise Exception("Unexpected Image Shape: %s" str(image_data.shape))
			dataset[num_images,:,:]=image_data
			num_images+=1
		except IOError as e:
			print('Could not read:',image_file,':',e,'- it\'s ok, skipping.')
		dataset=datasetp[0:num_images,:,:]
		if num_images<min_num_images:
			raise Exception('Many fewer images than expected: %d < %d '%(num_images,min_num_images))
		print('Full dataset tensor: ',dataset.shape)
		print('Mean: ',np.mean(dataset))
		print('Standard deviation: ',np.std(dataset))
		return	dataset


	


