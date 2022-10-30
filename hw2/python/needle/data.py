import numpy as np
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
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

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            return np.flip(img, axis=1)
        else:
            return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        img_t = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), constant_values=(0))
        return img_t[self.padding + shift_x : self.padding + shift_x + img.shape[0], self.padding + shift_y: self.padding + shift_y + img.shape[1]]
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        if self.shuffle:
            t = np.arange(len(self.dataset))
            np.random.shuffle(t)
            self.ordering = np.array_split(t, range(self.batch_size, len(self.dataset), self.batch_size))
        # for i in range(len(self.ordering) - 1):
        for i in self.ordering:
            out = list(self.dataset[i])
            for j in range(len(out)):
                out[j] = Tensor(out[j])
            yield out
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        return next(self.__iter__())
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.image_filename = image_filename
        self.label_filename = label_filename
        self.transforms = transforms
        self.image_data, self.label_data = parse_mnist(self.image_filename, self.label_filename)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        x = self.image_data[index]
        y = self.label_data[index]
        if len(x.shape) == 1:
            x = x.reshape(28, 28, 1)
        else:
            x = x.reshape(x.shape[0], 28, 28, 1)
        if self.transforms is not None:
            # x = x.reshape((28, 28, 1))
            if len(x.shape) == 3:
                for t in self.transforms:
                    x = t(x)
            else:
                for i in range(x.shape[0]):
                    for t in self.transforms:
                        x[i] = t(x[i])
        # x = x.reshape(1,28,28,1)
        return x, y
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.image_data)
        ### END YOUR SOLUTION

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
