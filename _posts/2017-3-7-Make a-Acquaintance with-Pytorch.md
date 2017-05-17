Recently, Facebook has opened a new deep learning framework called __PyTorch__, which puts python first and is in an early-release Beta. I used __Keras__ before and want to know what is the differences between Keras and PyTorch. This article will let you and me make a acquaintance with __PyTorch__.

In this article, I use PyTorch to build a convolutional neural network to recognize image in CIFAR-10 dataset. [Here](https://github.com/pytorch/tutorials/blob/master/Deep%20Learning%20with%20PyTorch.ipynb
) is the offical tutorial about learning/using PyTroch.

Notes:
- If you don't have PyTorch on your computer, you can download and intall it from [offical github pages](https://github.com/pytorch/pytorch#installation).

- If you don't have CIFAR-10 dataset, you can download it from [here](https://www.cs.toronto.edu/~kriz/cifar.html).

## Main Concepts
Here are some main concepts about PyTorch, it will help you undertand how PyTorch works and how to write your own neural nets.
- _Tensors_<br>
    PyTorch offers most tensor operations, including transposing, indexing, slicing, mathematical operations, linear algebra, random numbers, etc. `Tensors` in PyTorch are similar to numpy's ndarrays, with adding accelerate computing on GPU. You can define a tensor via
        ```python
        import torch
        # construct a 3x3 matrix without initialised
        x = torch.Tensor(5, 3)
        ```

- _Numpy Bridge_<br>
    As we all know, numpy is an indispensable part of python for scientific computation and many data is constructed as numpy. Pytorch provide an easy to convert between tensors and numpy. It's really convenient!
    - Convert PyTorch tensor into numpy array, E.g.
        ```python
        a = torch.ones(3)
        b = a.numpy()
        ```
    - Convert numpy array into PyTorch tensor
        ```python
        a = np.ones(5)
        b = torch.from_numpy(a)
        ```

- _Cuda Tensors_<br>
    If you want to calculate on GPU, you can use:
        ```python
        # let us run this cell only if CUDA is available
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
            x + y
        ```
    Because there is no high-performance GPU on my computer, I have not yet try this...

- _Autograd_<br>
    `Autograd` means automatic differentiation, provides automatic differentiation for all operations on tensors. You need to understand it with another concept _Variable_ together.

- _Variable_<br>
    `Variable` is used to wrap a tensor and records the operations applied to it. In my view, I regrad it as a _Symbol_ represents tensor and is uesd during computing, especially for differentiation.

- _Function_<br>
    `Function` is used to define formulas for differentiating operations on _Variable_.

## Tips
### Trick of making batch of training data with `torch.utils.data`
Suppose that you have trainning data `x_train` and `y_train` which you load them via `numpy`. You wan to train your neural network with `batch_size=64`. A convenient method is using `torch.utils.data`,<br>
1. Convert `x_train` and `y_train` into `tensor`
    ```python
    import torch
    import torch.utils.data as utils_data
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    ```

2. Build data loader
    ```python
    training_samples = utils_data.TensorDataset(x, y)
    data_loader = utils_data.DataLoader(training_samples, batch_size=64)
    ```

3. Using data loader like followings during training
    ```python
    import torch.autograd.Variable as Variable

    def train():
        for epoch in range(10):
            for i, data in enumerate(data_loader):
                # Get each batch
                inputs, labels = data
                # Convert tensor into Variable
                inputs, labels = Variable(inputs), Variable(labels)
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward + backward + optimize
                outputs = neural_network.forward(inputs)
                loss = loss_function(outputs, labels)
                loss.backwark()
                optimizer.step()    
    ```

- **Pay attention!**
    If training data is too large to read all into memory, it is a solution to read a batch from training data. Pytorch supports buid a `custom dataset` using `torch.utils.Dataset`.<br>
    Class `torch.utils.Dataset` is ([\*ref](http://pytorch.org/docs/_modules/torch/utils/data/dataset.html#Dataset))
    ```python
    class Dataset(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
    ```

    You need to rewrite `__getitem__` and `__len__` functions ([\*ref](https://discuss.pytorch.org/t/loading-huge-data-functionality/346/10)), the followings show a example:<br>
    ```python
    class LSPDataSet(utils_data.Dataset):
        def __init__(self, config):
            self.dataset_path = config['dataset_path']
            self.num_samples =config['train_data_size']
            self.ids_list = list(range(1, self.num_samples + 1))
            random.shuffle(self.ids_list)

        def __getitem__(self, index):
            image = Image.open('{}train/{:>06}.png'.format(self.dataset_path, self.ids_list[index]))
            image = np.array(image)
            image = np.rollaxis(image, 2, 0)
            label = np.load('{}train_label(pytorch)/{:>06}.npy'.format(self.dataset_path, self.ids_list[index]))
            image = np.array(image).astype(np.float32)
            label = np.array(label).astype(np.int)
            return image, label

        def __len__(self):
            return len(self.ids_list)
    ```

- Reference
    - [Pytorch doc: torch.utils.data](http://pytorch.org/docs/data.html)
    - [Stackoverflow: PyTorch How to use DataLoader for custom Datasets](http://stackoverflow.com/questions/41924453/pytorch-how-to-use-dataloaders-for-custom-datasets)
    - [PyTorch Forums: Loading huge data functionality](https://discuss.pytorch.org/t/loading-huge-data-functionality/346)

## Let's Have a Try
If we want to build a neural net for recognize image from CIFAR-10, what should we do?
1. Prepare data<br>
    We need to make our data suitable for PyTorch inputs. PyTorch provide a package called `torchvision` which offers some common datasets, you can get more details from [here](http://pytorch.org/docs/torchvision/torchvision.html).

2. Design your neural network<br>
    Here are a example network from [Offical Tutorial](https://github.com/pytorch/tutorials/blob/master/Deep%20Learning%20with%20PyTorch.ipynb)

    ```python
    from torch.autograd import Variable
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import torch

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool  = nn.MaxPool2d(2,2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1   = nn.Linear(16*5*5, 120)
            self.fc2   = nn.Linear(120, 84)
            self.fc3   = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16*5*5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()
    ```
3. Define loss function and optimizer
    ```python
    criterion = nn.CrossEntropyLoss() # use a Classification Cross-Entropy loss
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    ```
4. Train your network
    ```python
    for epoch in range(2): # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()        
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 2000 == 1999: # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')
    ```

## Problems and Solution:
- Cannot download datasets fluentlly.<br>
    Download CIFAR-10 from [here](https://www.cs.toronto.edu/~kriz/cifar.html). The script provided by the website for loading data is just for python2.7, if you want to use python3+, you need to replace `import Cpickle` by `import pickle` or `from _pickle import Cpickle`. And then you need to write some codes to preprocess the data to fit PyTorch input requires. I write a script to do this and save it also with `pickle`, you can get it [here](还没有上传！).

- `KeyError: <class 'torch.ByteTensor'>` occurs when training.<br>
    I met this error when I did training first time. My training data is loaded as numpy and it saved as `np.int`, and when it is transformed into PyTorch tensor, it is just a `torch.CharTensor`, but in PyTorch most operations are only defined on FloatTensor and DoubleTensor (and cuda.HalfTensor)[ref][ref_1]. So it will occur the error. The solution is using `numpy.astype(np.float32/np.float64)` to convert int into float.<br>
    [Here][ref_2] are some torch equivalents of numpy functions, maybe useful.

- Error `optimizing a parameter that doesn't require gradients`
    Solution:<br>
    `optimizer.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)`
    - Reference
        - [pytorch github #679: Allow optimizers to skip nn.Parameters that have requires_grad=False](https://github.com/pytorch/pytorch/issues/679)

This article is not abundant for using PyTorch, but still wish it would be useful for you and let us progress together:thumbsup:!

[ref_1]: https://discuss.pytorch.org/t/bytetensor-not-working-with-f-conv2d/802/2
[ref_2]: https://github.com/torch/torch7/wiki/Torch-for-Numpy-users
