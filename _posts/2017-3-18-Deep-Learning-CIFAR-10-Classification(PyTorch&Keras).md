This article will introduce how build a convolutional neural network by using python and popular dl framework.

Here we will build two network versions, one is by _PyTorch_ and another is by _Keras_. Then we train the networks on CIFAR-10 dataset.

# Requirements
- python
- CIFAR-10 dataset
- PyTorch
- Keras
- Some useful python packages:
    - numpy
    - pickle
    - sklearn
    - opencv
    - yaml

# Preprocess
At first, we need to preprocess the CIFAR-10 dataset, you can download it from [here](https://www.cs.toronto.edu/~kriz/cifar.html). Another way to get the dataset is from PyTorch or keras built-in utilities for downloading datasets. Here we download it from offical website and preprocess it.

1. Tidy dataset
    The raw CIFAR-10 data is saved as a type of python `dict`,  we will use `pickle` to read and extract the image data and labels. I have written a script to do this and you can get it from [here](). The key codes are shown as below.
    ```python
    save_data = {'data': [], 'label': []}
    for file_id in range(1, 6):
        file = 'F:/Windfall/Common Dataset/cifar-10-batches-py/data_batch_{}'.format(
            file_id)
        data = unpickle(file)
        for i in range(len(data['labels'])):
            image_data = np.array(data['data'][i])
            r = image_data[0:1024].reshape(32, 32)
            g = image_data[1024:2048].reshape(32, 32)
            b = image_data[2048:].reshape(32, 32)
            #
            sample_data = np.array([r, g, b])
            label = data['labels'][i]
            #
            save_data['data'].append(sample_data)
            save_data['label'].append(label)
    ```
    You need to change `file` to be the same with your dataset path.

2. Normalize
    Each image has 3 channels, 32 * 32 pixels and the depth is 8 bits(0 ~ 255). It is strongly recommended to do normalizing. Here are two ways you can choose:
    - Max-Min Normalization
        $$
        x^* = \frac{x_i - min(x)}{max(x) - min(x)}
        $$
        In CIFAR-10 image, $max(x) = 255$ and $min(x) = 0$

    - Z-Score Normalization
        $$
        x^* = \frac{x - \mu}{\sigma}
        $$
        where, $\mu$ is mean value and $\sigma$ is standard deviation.

# PyTorch Version
PyTorch, created by Facebook, is a new member of Machine Learning/Deep Learning framework group.<br>
You can learn the basics of PyTorch from my blog or offical tutorial.
Here we will build a 3 layers 2D neural network using PyTorch.
1. Design network.
    ```python
    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    ```
2. Choose optimizer
    ```python
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
    ```
3. Train
    ```python
    def train(self, x_train, y_train):
        for epoch in range(int(self.config['n_epoch'])):
            running_loss = 0.0
            # Train each batch
            for batch_id in range(self.n_batch):
                # Batch samples id.
                batch_samples_id = self.samples_id[
                    batch_id * self.batch_size:(batch_id + 1) * self.batch_size]
                # Get the inputs.
                inputs = torch.from_numpy(x_train[batch_samples_id])
                labels = torch.from_numpy(y_train[batch_samples_id])
                # Wrap them in Variable.
                inputs, labels = Variable(inputs), Variable(labels)
                # Zero the parameter gradients.
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                # Update statistics
                running_loss += loss.data[0]
            # Print
            print('\nepoch={}, loss={:.10f}\n'.format(
                epoch + 1, running_loss / (batch_id * len(x_train))))
            running_loss = 0.0
        print('Finished Training')
        if self.config['save_trained_model'] is True:
            print('Saving trained model')
            torch.save(self.net.state_dict(), self.config[
                       'save_trained_model_path'])
            print('Finished saving')
    ```
4. Test your net work
    ```python
    for i, sample in enumerate(x_predict):
            sample = torch.from_numpy(np.array([sample]).astype(np.float32))
            sample = Variable(sample)
            y = trained_cnn_net(sample)
            _, y = (torch.max(y, 1))
            y = y.data.numpy()[0][0]
            y_predict.append(y)
        return y_predict
    ```
This network has 3 convolutional layers, the accuracy on test dataset is 70%.

# Keras Version
In keras version, we add the number of convolutional layers and kernels and the the accuracy on test dataset is 80%.
1. Design network
    ```python
    def build_net(self):
        self.net = Sequential()

        self.net.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, 32, 32)))
        self.net.add(Activation('relu'))
        self.net.add(Convolution2D(32, 3, 3))
        self.net.add(Activation('relu'))
        self.net.add(MaxPooling2D(pool_size=(2, 2)))
        self.net.add(Dropout(0.25))

        self.net.add(Convolution2D(64, 3, 3, border_mode='same'))
        self.net.add(Activation('relu'))
        self.net.add(Convolution2D(64, 3, 3))
        self.net.add(Activation('relu'))
        self.net.add(MaxPooling2D(pool_size=(2, 2)))
        self.net.add(Dropout(0.25))

        self.net.add(Flatten())
        self.net.add(Dense(512))
        self.net.add(Activation('relu'))
        self.net.add(Dropout(0.5))
        self.net.add(Dense(10))
        self.net.add(Activation('softmax'))
    ```
2. Compile network
    ```python
    def compile_net(self):
        self.net.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        self.net.summary()
    ```
3. Train
    ```python
    def train(self, x_train, y_train):
        y_train = y_train.reshape(-1, 1)
        y_train = self.onehot_encode(y_train)
        self.net.fit(x_train, y_train,
                     batch_size=self.config['batch_size'],
                     nb_epoch=self.config['n_epoch'],
                     validation_split=0.01,
                     shuffle=True)
        # Save trained model
        if self.config['save_trained_model'] is True:
            self.net.save(self.config['save_trained_model_path'])
    ```
4. Predict
    ```python
    def predict(self, x_predict):
        if self.config['load_trained_model'] is True:
            self.net = keras.models.load_model(self.config['trained_model_path'])
            n_samples = len(x_predict)
            y_predict = self.net.predict_classes(x_predict)
            return y_predict
    ```
# Note
Since I was recently busy with the lab project and have no enough time to make this article more detailed, I will modificate in the future. If you want to get source code and have a test, you can get source code from my [github repository.](https://github.com/rivergold/Deep-Learning-CIFAR-10-Classification)

<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.7.1/katex.min.css" integrity="sha384-wITovz90syo1dJWVh32uuETPVEtGigN07tkttEqPv+uR2SE/mbQcG7ATL28aI9H0" crossorigin="anonymous">
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.7.1/katex.min.js" integrity="sha384-/y1Nn9+QQAipbNQWU65krzJralCnuOasHncUFXGkdwntGeSvQicrYkiUBwsgUqc1" crossorigin="anonymous"></script>
