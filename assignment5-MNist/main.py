# ref https://github.com/ChawDoe/LeNet5-MNIST-PyTorch/blob/master/train.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

########################################## Lenet-5 Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.fc2(out)
        out = self.relu4(out)
        out = self.fc3(out)
        out = self.relu5(out)
        return out


###########################################
if __name__ == "__main__":
    batch_size = 64
    mnist_train = mnist.MNIST(
        "./train", train=True, download=True, transform=ToTensor()
    )
    mnist_test = mnist.MNIST("./test", train=False, download=True, transform=ToTensor())
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    net = Net()
    # The cost function we used for logistic regression
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    n_epochs = 938
    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        for imgs, labels in train_loader:
            optimizer.zero_grad()
            outputs = net(imgs)
            train_loss = loss_fn(outputs, labels)
            train_loss.backward()
            optimizer.step()
        if epoch == 1 or epoch % 10 == 0:
            print("Epoch{}, Training loss{}".format(epoch, train_loss))

    # Test Section
    import sys
    from PIL import Image

    mnist_sample = mnist.MNIST("./sample", train=False, download=True, transform=None)
    for j in range(100):
        for i in range(10):
            if mnist_sample[j][1] == 0:
                mnist_sample[j][0].save("zero.png")
            elif mnist_sample[j][1] == 1:
                mnist_sample[j][0].save("one.png")
            elif mnist_sample[j][1] == 2:
                mnist_sample[j][0].save("two.png")
            elif mnist_sample[j][1] == 3:
                mnist_sample[j][0].save("three.png")
            elif mnist_sample[j][1] == 4:
                mnist_sample[j][0].save("four.png")
            elif mnist_sample[j][1] == 5:
                mnist_sample[j][0].save("five.png")
            elif mnist_sample[j][1] == 6:
                mnist_sample[j][0].save("six.png")
            elif mnist_sample[j][1] == 7:
                mnist_sample[j][0].save("seven.png")
            elif mnist_sample[j][1] == 8:
                mnist_sample[j][0].save("eight.png")
            elif mnist_sample[j][1] == 9:
                mnist_sample[j][0].save("nine.png")
    if len(sys.argv) != 2:
        print(
            "<Usage>: Input value is not coming ex)main.py one.png "
        )  # program이나py파일이 하나라도 안들어왔을 경우 exception 처리
        exit(1)
    image_ = Image.open(sys.argv[1])
    image = torch.zeros(16, 1, 28, 28)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # image to Tensor
        ]
    )
    image = image + transform(image_)
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    optimizer.zero_grad()
    # forward propagation
    model_output = net(image)
    print(f"Assume : {model_output.argmax(dim=1)[0]}")
