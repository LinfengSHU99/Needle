import math
import struct
import gzip
import numpy as np

import sys
sys.path.append('python/')
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    images_content = gzip.open(image_filesname).read()
    labels_content = gzip.open(label_filename).read()
    feature_size = struct.unpack(">ii", images_content[8:16])
    image_num = struct.unpack(">i", images_content[4:8])
    images_data = struct.unpack(">" + "B" * len(images_content[16:]), images_content[16:])
    labels_data = struct.unpack(">" + "B" * len(labels_content[8:]), labels_content[8:])
    images_data = np.array(images_data, dtype=np.float32)
    range_ = images_data.max() - images_data.min()
    images_data = (images_data - images_data.min()) / range_
    images_data = images_data.reshape((image_num[0], feature_size[0] * feature_size[1]))
    labels_data = np.array(labels_data, dtype=np.uint8)
    return images_data, labels_data
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    t = ndl.log((ndl.exp(Z)).sum(axes=1)) - (Z * y_one_hot).sum(axes=1)
    return t.sum() / t.shape[0]
    # return ndl.mean(ndl.log((ndl.e ** Z).sum(axis=1)) - Z[ndl.arange(0, Z.shape[0]), y])
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION

    index = np.arange(0, X.shape[0], batch)
    if index[-1] < X.shape[0]:
        index = np.append(index, X.shape[0])
    for i in range(len(index) - 1):
        x_batch = ndl.Tensor(X[index[i]: index[i + 1]])
        y_batch = y[index[i]: index[i + 1]]
        I_y = np.zeros((batch, W2.shape[1]))
        I_y[np.arange(0, batch), y_batch] = 1
        I_y_tensor = ndl.Tensor(I_y)
        loss = softmax_loss(ndl.relu(x_batch @ W1) @ W2, I_y_tensor)
        loss.backward()
        W1.cached_data -= lr * W1.grad.numpy()
        W2.cached_data -= lr * W2.grad.numpy()
    return W1, W2
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)