import numpy as np
import chainer

import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import Variable
from chainer import serializers
import math
import utilities as util 
import matplotlib 
import matplotlib.pyplot as plt 
import os
import sys 

import cifar10_extract as cifar
from Map import Map

constants = Map({})
constants.CURRENT_WORKING_DIRECTORY = os.getcwd()


parameters = Map({})
parameters.USE_TANH_ACTIVATION_FUNCTION = False
parameters.NZ = 100
parameters.ALPHA = 0.0002
parameters.BETA1 = 0.5
parameters.WEIGHT_DECAY = 0.00001
parameters.NUM_CHANNELS = 3
parameters.IMAGE_HEIGHT = 32
parameters.IMAGE_WIDTH = 32
parameters.NUM_EPOCH = 100
parameters.NSAMPLES = 16 
parameters.BATCH_SIZE = 100
parameters.PREPROCESSED_DATA_FILE = constants.CURRENT_WORKING_DIRECTORY  + "/preprocessed_cifar10.npy"
parameters.IMAGE_SAVING_INTERVAL = 1000
parameters.GPU_ID = 21
parameters.SAVED_ZVIS = "zvis.npy"

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def invert_preprocess(x):
    return (x+1)/2

def visualize_cifar10_images(zvis, img_name):

    """
        Visualize the set of images having shape (parameters.BATCH_SIZE, height*width*num_channels)
    """
    nsamples = zvis.shape[0]

    z = zvis

    # resample half of the z 
    z[nsamples/2:,:] = (xp.random.uniform(-1, 1, (nsamples/2, parameters.NZ)).astype(np.float32))

    z = Variable(z)
    x = gen(z, test=True)
    yo = x.data

    #yo = model.sample(z).data
    #util.checkAllNonNegative(yo, "yo", "before invert transform in sampling ")
    
    yo = invert_preprocess(yo)
    yo = xp.reshape(yo, (yo.shape[0], parameters.NUM_CHANNELS*parameters.IMAGE_WIDTH*parameters.IMAGE_HEIGHT))                    

    # why?
    # Basically brings the image pixels in the range of 0 and 1
    # why?
    #yo = invert_preprocessing(yo)
    
    #util.checkAllNonNegative(yo, "yo", "after invert transform in sampling ")
   
    #X = xp.reshape(model.sample(z), (parameters.NSAMPLES, 32, 32, 3), order='F') # This in imho is Buggy. This leads to vertical inversion, w -> h instead of w -> w

    X = np.empty((nsamples, parameters.IMAGE_HEIGHT, parameters.IMAGE_WIDTH, parameters.NUM_CHANNELS))
    for i in range(nsamples):
        for j in range(3072):
            X[i,(j%1024)/32,(j%1024)%32,j/1024] = yo[i,j ]

    #X = (model.sample(z).data).reshape((parameters.NSAMPLES, 32, 32, 3), order='F')
    #print X

    #X = xp.reshape(model.sample(z), (parameters.NSAMPLES, 32, 32, 3), order='F') # This in imho is Buggy. This leads to vertical inversion, w -> h instead of w -> w

    X/=X.max()

    visualize_image_grid(X, 4, 4, 16, 16, img_name)


def visualize_image_grid(inp, n_rows, n_cols, fig_width, fig_height, img_name):
    f, axarr = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    nsamples = n_rows * n_cols
    for i in range(nsamples):
        axarr[i / n_cols, i % n_cols].imshow(inp[i,:, :, : ])

    f.savefig(img_name)
    plt.close(f)

def saveLinePlot(num_list, effective_epoch, i, dir_path):
    x = range(1, len(num_list) + 1)
    y = num_list
    plt.plot(x,y)
    
    img_name = dir_path + "/loss" + "_epoch_" + str(effective_epoch) + "_" + str(i) + ".png"
    plt.savefig(img_name)
    plt.close()    

class Generator(chainer.Chain):
    
    def __init__(self, nz =100):
        super(Generator, self).__init__(
            
            l0z = L.Linear(nz, 4*4*512, wscale=0.02*math.sqrt(nz)),
            #deconv upsamples spatially
            dc1 = L.Deconvolution2D(512, 256, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*512)),
            dc2 = L.Deconvolution2D(256, 64, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*256)),
            # SCALE: can try using 128 instead of 64
            dc3 = L.Deconvolution2D(64, 3, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*64)),
            # the output of this deconvolution is (batch_size, 3 x 32 x 32) image
            bn0l = L.BatchNormalization(4*4*512),
            bn1 = L.BatchNormalization(256),
            bn2 = L.BatchNormalization(64),

        )
        
    def __call__(self, z, test=False):

        h = F.reshape(F.relu(self.bn0l(self.l0z(z))), (z.data.shape[0], 512, 4, 4 ))
        h = F.relu(self.bn1(self.dc1(h), test=test))
        h = F.relu(self.bn2(self.dc2(h), test=test))
        
        #print "checking if before dc3 is all non negative", (h.data>=0).all()
        
        if(parameters.USE_TANH_ACTIVATION_FUNCTION):
            h = F.tanh(self.dc3(h))
        else:
            h = (self.dc3(h))
        print h.shape
        return h


class Discriminator(chainer.Chain):
    
    def __init__(self):
        super(Discriminator, self).__init__(

            c1 = L.Convolution2D(3, 64, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*3)),
            c2 = L.Convolution2D(64, 256, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*64)),
            c3 = L.Convolution2D(256, 512, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*256)),
            l0z = L.Linear(4*4*512, 10, wscale=0.02*math.sqrt(4*4*512)),
            bn1 = L.BatchNormalization(64),
            bn2 = L.BatchNormalization(256),
            bn3 = L.BatchNormalization(512),

        )
        
    def __call__(self, z, test=False):
        # mattya's implementation does not have bn after c1
        h = F.elu(self.bn1(self.c1(z), test=test))
        h = F.elu(self.bn2(self.c2(h), test=test))
        h = F.elu(self.bn3(self.c3(h), test=test))
        h = self.l0z(h)
        print h.shape
        return h

def preprocess(data):
    data = data.astype(np.float32)
    data = (data-127.5)/127.5
    return data 


def get_latest_epoch_trained_before(base_name, dir_name):
    """
        Returns the count of epochs trained before. 
    """
    ct = 0 
    # the base name for the serialized optimizer is "dcgan_state_gen_", which also has 
    # same length. So simply using one of them

    
    num_chars_before_idx = len(base_name)
    for f in os.listdir(dir_name):
        if f.endswith(".h5"):
            dot_idx = f.find('.')
            num = int(f[num_chars_before_idx:dot_idx])
            ct = max(ct, num)
        else:
            continue
    return ct

def train_mattya_gan(gen, disc, epoch_idx_done_before=-1):
    # Train a gan model that uses an architecture inspired from https://github.com/mattya/chainer-DCGAN/blob/master/DCGAN.py
    # The epoch_idx_done_before parameter allows to start training from a previous checkpoint

    if(len(sys.argv)>1):
        prev_exp_dir = sys.argv[1]
        print "prev exp", prev_exp_dir
        print "cwd", constants.CURRENT_WORKING_DIRECTORY
        serialized_model_dir = constants.CURRENT_WORKING_DIRECTORY + "/" + prev_exp_dir + "/models" 
        base_name = "dcgan_model_gen_epoch=" 
        epoch_idx_done_before = get_latest_epoch_trained_before(base_name, serialized_model_dir)    
        
    o_gen = optimizers.Adam(alpha = parameters.ALPHA, beta1 = parameters.BETA1)
    o_dis = optimizers.Adam(alpha = parameters.ALPHA, beta1 = parameters.BETA1)
    o_gen.setup(gen)
    o_dis.setup(disc)

    o_gen.add_hook(chainer.optimizer.WeightDecay(parameters.WEIGHT_DECAY))
    o_dis.add_hook(chainer.optimizer.WeightDecay(parameters.WEIGHT_DECAY))
    

    if(epoch_idx_done_before>=0):
        print "epochs done before are", epoch_idx_done_before
        serializers.load_hdf5(serialized_model_dir+"/dcgan_model_gen_epoch=" + str(epoch_idx_done_before) + ".h5", gen)
        serializers.load_hdf5(serialized_model_dir+"/dcgan_state_gen_epoch=" + str(epoch_idx_done_before) + ".h5", o_gen)
        serializers.load_hdf5(serialized_model_dir+"/dcgan_model_dis_epoch=" + str(epoch_idx_done_before) + ".h5", dis)
        serializers.load_hdf5(serialized_model_dir+"/dcgan_state_dis_epoch=" + str(epoch_idx_done_before) + ".h5", o_dis)
        constants.CURRENT_WORKING_DIRECTORY += "/" + prev_exp_dir
        zvis_file = constants.CURRENT_WORKING_DIRECTORY +  parameters.SAVED_ZVIS
        zvis = np.load(zvis_file)
    else:    
        
        timestamp = util.get_current_time_stamp()
        constants.CURRENT_WORKING_DIRECTORY += "/Exp_" + timestamp
        mkdir(constants.CURRENT_WORKING_DIRECTORY)
        zvis = xp.random.uniform(-1, 1, (parameters.NSAMPLES, parameters.NZ)).astype(np.float32)
        zvis_file = constants.CURRENT_WORKING_DIRECTORY + "/" + parameters.SAVED_ZVIS
        np.save(zvis_file, zvis)

    preprocessed_data_path = parameters.PREPROCESSED_DATA_FILE

    if(not os.path.isfile(preprocessed_data_path)):
        train_data = cifar.load_cifar10()
        train_data = preprocess(train_data) 
        np.save(preprocessed_data_path, train_data)
    else:
        train_data = np.load(preprocessed_data_path)

    n_train = train_data.shape[0]

    img_path = constants.CURRENT_WORKING_DIRECTORY + "/images"   
    mkdir(img_path)
    
    out_model_dir = constants.CURRENT_WORKING_DIRECTORY + "/models"
    mkdir(out_model_dir)
    
    gen_loss_path = constants.CURRENT_WORKING_DIRECTORY + "/gen_loss"   
    mkdir(gen_loss_path)
    
    dis_loss_path = constants.CURRENT_WORKING_DIRECTORY + "/dis_loss"   
    mkdir(dis_loss_path)
    
    gen_loss_list=[]
    dis_loss_list=[]
    
    for epoch in xrange(epoch_idx_done_before+1, parameters.NUM_EPOCH):
        sum_dis_loss = np.float32(0)
        sum_gen_loss = np.float32(0)
        
        np.random.shuffle(train_data)

        for i in xrange(0, n_train, parameters.BATCH_SIZE):
            # 0: from dataset   
            # 1: from noise 

            # train generator
            z = Variable(xp.random.uniform(-1, 1, (parameters.BATCH_SIZE, parameters.NZ)).astype(np.float32))
            x = gen(z)
            yl = dis(x)

            L_gen = F.softmax_cross_entropy(yl, Variable(xp.zeros(parameters.BATCH_SIZE).astype(np.int32)))
            L_dis = F.softmax_cross_entropy(yl, Variable(xp.ones(parameters.BATCH_SIZE).astype(np.int32)))
            
            # train discriminator
            
            x2 = train_data[i:i+parameters.BATCH_SIZE] 
            x2 = xp.reshape(x2, (x2.shape[0], parameters.NUM_CHANNELS, parameters.IMAGE_WIDTH, parameters.IMAGE_HEIGHT))                    
            x2 = Variable(xp.asarray(x2).astype(np.float32))

            yl2 = dis(x2)
            L_dis += F.softmax_cross_entropy(yl2, Variable(xp.zeros(parameters.BATCH_SIZE).astype(np.int32)))


            o_gen.zero_grads()
            L_gen.backward()
            o_gen.update()
            
            o_dis.zero_grads()
            L_dis.backward()
            o_dis.update()
            
            curr_batch_gen_loss = L_gen.data 
            sum_gen_loss += curr_batch_gen_loss
            curr_batch_dis_loss = L_dis.data
            sum_dis_loss += curr_batch_dis_loss 

            gen_loss_list.append(curr_batch_gen_loss/parameters.BATCH_SIZE)
            dis_loss_list.append(curr_batch_dis_loss/parameters.BATCH_SIZE)
            
            if(i%parameters.IMAGE_SAVING_INTERVAL==0):
                name = "out_epoch=" + str(epoch) + "_batch=" + str(i)
                img_name = img_path + "/" + name + ".png"

                visualize_cifar10_images(zvis, img_name)
                saveLinePlot(gen_loss_list, epoch, i, gen_loss_path)
                saveLinePlot(dis_loss_list, epoch, i, dis_loss_path)
                
        serializers.save_hdf5("%s/dcgan_model_dis_epoch=%d.h5"%(out_model_dir, epoch),dis)
        serializers.save_hdf5("%s/dcgan_model_gen_epoch=%d.h5"%(out_model_dir, epoch),gen)
        serializers.save_hdf5("%s/dcgan_state_dis_epoch=%d.h5"%(out_model_dir, epoch),o_dis)
        serializers.save_hdf5("%s/dcgan_state_gen_epoch=%d.h5"%(out_model_dir, epoch),o_gen)
        print 'epoch end', epoch, sum_gen_loss/n_train, sum_dis_loss/n_train


# xp = cuda.cupy
xp = np 
# cuda.get_device(parameters.GPU_ID).use()

gen = Generator()
dis = Discriminator()
#gen.to_gpu()
#dis.to_gpu()

train_mattya_gan(gen, dis)