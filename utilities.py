import os
import sys
import numpy as np 
import chainer
import chainer.computational_graph as cg
import matplotlib.pylab as plt
import datetime

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def write_log(path, filename):
    mkdir(path)
    f = file(path + filename, 'w')
    sys.stdout = f


def get_current_time_stamp():
    return "{:%Y-%m-%d_%H:%M:%S}".format(datetime.datetime.now())
    
def checkAllNonNegative(arr, arr_name, extra=""):
	print "checking if all elements of array ", arr_name, " are all non_negative : ", extra
	print (arr>=0).all()

def make_computational_graph(var_list, dir, name):
    g = cg.build_computational_graph(var_list)
    with open(dir + name, 'w') as o:
        o.write(g.dump())

def debug_chainer_variable(var, name):
    print "checking for variable named ", name
    print "shape of it is ", var.data.shape
    #print "data for it is ", var.data
    print "max element is ", (var.data).max()
    print "min element is ", (var.data).min()
    print "mean element is ", (var.data).mean()
    print "median element is ", np.median(xp.asnumpy(var.data).astype(np.float32))

def visualize_distance_matrix(mat, direc, name):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.imshow(mat, interpolation='nearest', cmap=plt.cm.ocean)
    plt.colorbar()
    timestamp = "{:%Y-%m-%d_%H:%M:%S}".format(datetime.datetime.now())
    plt.savefig(direc+name + str(timestamp))
    plt.close()

def visualize_lambda_m(mat, direc, name):
    flat_mat = np.reshape(mat, (mat.shape[0]*mat.shape[1], ))
    plt.hist(flat_mat, bins='auto')
    timestamp = get_current_time_stamp()
    plt.savefig(direc +name + str(timestamp))
    plt.close()

def convertGrayscaleToRGB(mat):
    rgb = np.empty((mat.shape[0], 3*mat.shape[1]))
    rgb[:, 0:mat.shape[1]] = mat[:, 0:mat.shape[1]]
    rgb[:, mat.shape[1]:2*mat.shape[1]] = mat[:, 0:mat.shape[1]]
    rgb[:, 2*mat.shape[1]:3*mat.shape[1]] = mat[:, 0:mat.shape[1]]
    
    return rgb
