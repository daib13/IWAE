# IWAE

This is a re-implementation of importance weighted autoencoders (IWAE)[1] in tensorflow.
I will try to improve IWAE later.

## Demonstrate the regularization effect of IWAE

Useage: python iwae.py K ALPHA GPU_ID

K: The parameter K in Burda et al.[1]. The number of samples.

ALPHA: A samll value to avoid log(0). A typical choice is 0.01

GPU_ID: The gpu you want to use.

It trains a IWAE on a synthesized dataset. The dataset is generated using a 2-layer neural network with fixed random weights. Thus it lies on a two dimensional manifold. The manifold dimension is set to be 20. A IWAE is trained on the synthesized dataset with the latent dimension to be 30, which is larger than the groundtruth value. We check the active dimension that IWAE learned. If the regularization effect of IWAE works well, the active dimension should be 20.

[1]Burda, Yuri, Roger Grosse, and Ruslan Salakhutdinov. "Importance weighted autoencoders." arXiv preprint arXiv:1509.00519 (2015).
