import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    m = nn.Sequential(
        nn.Linear(in_features=dim, out_features=hidden_dim),
        norm(dim=hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim=dim),
    )
    ret = nn.Sequential(nn.Residual(m), nn.ReLU())
    return ret
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    lst = []
    linear = nn.Linear(dim, hidden_dim)
    relu = nn.ReLU()
    for i in range(num_blocks):
        lst.append(ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob))
    linear2 = nn.Linear(hidden_dim, num_classes)
    return nn.Sequential(linear, relu, *lst, linear2)
    # ResidualBlock(1,1)
    # return nn.Sequential(
    #     nn.Linear(dim, hidden_dim),
    #     nn.ReLU(),
    #     # *lst,
    #     ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob),
    #     ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob),
    #     nn.Linear(hidden_dim, num_classes),
    # )
    ### END YOUR SOLUTION




def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    i = 0
    loss_lst = []
    err_lst = []
    if opt is not None:
        model.train()
        for x, y in dataloader:
            opt.reset_grad()
            i += x.shape[0]
            x = nn.Flatten()(x)
            pred = model(x)
            loss = nn.SoftmaxLoss()(pred, y)
            loss_lst.append(loss.detach().numpy())
            loss.backward()
            opt.step()
            logits = pred.detach().numpy()
            pred_label = np.argmax(logits, axis=1)
            err_lst.append((np.sum(pred_label != y.detach().numpy())))
        return sum(err_lst) / i, sum(loss_lst) / len(loss_lst)
    else:
        model.eval()
        for x, y in dataloader:
            i += x.shape[0]
            x = nn.Flatten()(x)
            pred = model(x)
            loss = nn.SoftmaxLoss()(pred, y)
            loss_lst.append(loss.detach().numpy())
            logits = pred.detach().numpy()
            pred_label = np.argmax(logits, axis=1)
            err_lst.append((np.sum(pred_label != y.detach().numpy())))
        return sum(err_lst) / i, sum(loss_lst) / len(loss_lst)
    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_img_filename = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
    train_label_filename = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    test_img_filename = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
    test_label_filename = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
    train_dataset = ndl.data.MNISTDataset(train_img_filename, train_label_filename)
    test_dataset = ndl.data.MNISTDataset(test_img_filename, test_label_filename)
    train_dataloader = ndl.data.DataLoader(train_dataset, batch_size, True)
    test_dataloader = ndl.data.DataLoader(test_dataset, batch_size)
    model = MLPResNet(28*28, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for i in range(epochs):
        err, loss = epoch(train_dataloader, model, opt)
    test_err, test_loss = epoch(test_dataloader, model)
    return (err, loss, test_err, test_loss)
    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")
