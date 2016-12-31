import cPickle
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt 

file_prefix ="/home/sidak/Documents/ML Related Stuff/Datasets/cifar-10-batches-py/data_batch_"

#for time being let us skip the test data set (10000 images)

def unpickle(file):
    
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_cifar10():

	dta = np.zeros(shape=(50000, 3072), dtype = np.float32)

	for i in range(1, 6):
		filename = file_prefix+str(i)
		dic = unpickle(filename)
		idx =10000*(i-1)
		# replace by numpy operation later
		#data[10000*(i-1):10000*i,:] = 
		for j in range(len(dic['data'])):
			dta[idx+j] = dic['data'][j]

	#print dta
	#print dta.shape
	return dta

def load_cifar10_with_labels():
	dta = np.zeros(shape=(50000, 3072))
	labels = np.zeros(shape=(50000,))

	for i in range(1, 6):
		filename = file_prefix+str(i)
		dic = unpickle(filename)
		idx =10000*(i-1)
		for j in range(len(dic['data'])):
			dta[idx+j] = dic['data'][j]
			labels[idx+j] = dic['labels'][j]
	#print dta
	#print dta.shape
	return dta, labels

def visualize_image_grid(inp, n_rows, n_cols, fig_width, fig_height, img_name):
    f, axarr = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    nsamples = n_rows * n_cols
    for i in range(nsamples):
        axarr[i / n_cols, i % n_cols].imshow(inp[i,:, :, : ])

    f.savefig(img_name)
    plt.close(f)


# test
dta = load_cifar10()
print dta.max()
print dta.min()
sample = dta[0:16,]

X = np.empty((16, 32, 32, 3))


for i in range(16):
    #X[i,] = np.transpose(sample[i,], (1,2,0))

    for j in range(3072):
 	   X[i,(j%1024)/32,(j%1024)%32,j/1024] = sample[i,j]

X/=255.0
visualize_image_grid(X, 4, 4, 16, 16, "./check7.png")

#print dta[0:1,]
