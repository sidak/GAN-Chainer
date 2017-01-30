# GAN-Chainer

This repo is an implementation of Generative Adversarial Networks (GANs) using the Chainer library. GANs represents a new framework 
for estimating generative models, in which two neural networks called the *generator* and the *discriminator* compete in a minimax
game. 

Before going ahead, it is important to understand what are generative models. Take the following example (which I read perhaps
on StackOverflow): Identifying if a given painting is of Monalisa or not, is the job of a discriminative model. On the other hand, 
making a painting of Monalisa is the job of a generative model. 

The GANs framework details an indirect approach to learning Generative Models, rather than using a loss function. (Actually 
formulating such a loss function is a hard problem in itself). Considering that we take these models to be Multi-layer Neural Networks,
the generator network is pitted against a discriminator network. 

## Designing the Generator Network
The architecture of the generator network is inspired from the DCGAN paper. The main guidelines that are followed are:

1. All convolutional network: replace pooling by strided convolutions. 
2. No fully connected layers at the end.
3. Applying Batch-Normalization.
4. Using ReLU activation except Tanh for the output layer.

--- TO BE CONTINUED ---




