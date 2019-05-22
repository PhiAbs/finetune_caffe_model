import caffe
import os
import sys
import numpy as np
import re

model_def = '/home/pussycat/ppgn/nets/caffenet_DL_PROJECT/caffenet_deploy.prototxt'
#model_def = '/home/pussycat/finetune_caffe_model/models/caffenet/deploy.prototxt'
model_weight = '/home/pussycat/finetune_caffe_model/models/caffenet/caffenet_train_snapshot/solver_iter_55000.caffemodel'


net = caffe.Net(model_def, model_weight, caffe.TEST)

caffe.set_device(0)
caffe.set_mode_gpu()


#mu = np.load('/home/pussycat/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy')
#mu = mu.mean(1).mean(1)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1)) 
#transformer.set_mean('data', mu) 
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

test_img_dir = '/home/pussycat/finetune_caffe_model/data/test/'

sub_dir_list = [x[0] for x in os.walk(test_img_dir)]

f = open('/home/pussycat/finetune_caffe_model/data/dataset_info/test.txt')

_true = 0.0
_false = 0.0

for line in f:
	
	line = str(line)

	img = line[ : -4]
	#print(img)
	target_neuron = int(line[-3 : -2])
	#print(target_neuron)
	img_path = os.path.join(test_img_dir, img)

	image = caffe.io.load_image(str(img_path)) 

	transformed_image = transformer.preprocess('data', image)

	net.blobs['data'].reshape(50,3, 227, 227) 

	net.blobs['data'].data[...] = transformed_image

	output = net.forward()

	output_prob = output['log'][0]

	#print(output_prob.argmax())

	#print(str(target_neuron) + ' ' +str( output_prob.argmax()))

	if target_neuron == output_prob.argmax():
		_true += 1
	else:
		_false += 1

print('Pussycat Accurary is ' + str(float(_true/(_true+_false))))

