import torch
import numpy as np
import gzip
import struct
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

def sofmax(z, y_one_hot):
    t = torch.log(torch.exp(z).sum(dim=1)) - (z * y_one_hot).sum(dim=1)
    return t.sum() / t.shape[0]

def nn_epoch(x, y, w1, w2, lr, batch):
    x_batch = x[: batch]
    y_batch = y[: batch]
    I_y = torch.zeros((batch, w2.shape[1]))
    lst = [i for i in range(batch)]

    I_y[tuple(lst), y_batch.numpy().tolist()] = 1
    loss = sofmax(torch.relu(x_batch @ w1) @ w2, I_y)
    print("torch loss", loss)
    loss.backward()
    print("torch grad", w1.grad, w2.grad)
    w1 = w1 - (lr * w1.grad)
    w2 = w2 - (lr * w2.grad)
    print(np.linalg.norm(w1.detach().numpy()))
    print(np.linalg.norm(w2.detach().numpy()))
# %%
# x = torch.from_numpy(np.array([[1,2,3], [4,5,6]]).astype(np.float64))
# y_one_hot = torch.from_numpy(np.array([[1,0], [0,1]]))
# y = torch.from_numpy(np.array([0, 1]))
# w1 = torch.from_numpy(np.array([[1,2], [1,3], [4,0]]).astype(np.float64))
# w2 = torch.from_numpy(np.array([[1,0], [1,2]]).astype(np.float64))
# w1.requires_grad = True
# w2.requires_grad = True
# #%%
# z = x @ w1 @ w2
# # z = torch.FloatTensor([[3, 0], [9, 0]], )
# # z.requires_grad = True
# t = torch.log(torch.exp(z).sum(axis=1)) - (z * y_one_hot).sum(axis=1)
# loss = t.sum() / 2
# loss.backward()
# print(loss)
# print(w1.grad,w2.grad)
np.random.seed(0)
X,y = parse_mnist("data/train-images-idx3-ubyte.gz",
                  "data/train-labels-idx1-ubyte.gz")
X = torch.from_numpy(X)
y = torch.from_numpy(y)
W1 = np.random.randn(X.shape[1], 100).astype(np.float32) / np.sqrt(100)
W2 = np.random.randn(100, 10).astype(np.float32) / np.sqrt(10)
W1 = torch.from_numpy(W1)
W2 = torch.from_numpy(W2)
W1.requires_grad = True
W2.requires_grad = True

nn_epoch(X, y, W1, W2, 0.2, 100)

# r = torch.from_numpy(np.array([[1,-1],[0,1]]).astype(np.float32))
# print(torch.relu(r))
# r.requires_grad = True
# loss = torch.relu(r).sum()
# loss.backward()
# print(loss)
# print(r.grad)