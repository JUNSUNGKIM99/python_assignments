# 참고문헌: https://github.com/inmoonlight/PyTorchTutorial/blob/master/01_CNN.ipynb

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image

# 0~9까지 숫자파일을 일괄적으로 생산
mnist = datasets.MNIST("./mnist_data", train=True, download=True)
for j in range(100):
    for i in range(10):
        if mnist[j][1] == 0:
            mnist[j][0].save("zero.png")
        elif mnist[j][1] == 1:
            mnist[j][0].save("one.png")
        elif mnist[j][1] == 2:
            mnist[j][0].save("two.png")
        elif mnist[j][1] == 3:
            mnist[j][0].save("three.png")
        elif mnist[j][1] == 4:
            mnist[j][0].save("four.png")
        elif mnist[j][1] == 5:
            mnist[j][0].save("five.png")
        elif mnist[j][1] == 6:
            mnist[j][0].save("six.png")
        elif mnist[j][1] == 7:
            mnist[j][0].save("seven.png")
        elif mnist[j][1] == 8:
            mnist[j][0].save("eight.png")
        elif mnist[j][1] == 9:
            mnist[j][0].save("nine.png")

transform = transforms.Compose(
    [
        transforms.ToTensor(),  # image to Tensor
        transforms.Normalize((0.1307,), (0.3081,)),  # image, label
    ]
)

batch_size = 64
trn_dataset = datasets.MNIST(
    "../mnist_data/", download=True, train=True, transform=transform
)

trn_loader = torch.utils.data.DataLoader(
    trn_dataset, batch_size=batch_size, shuffle=True
)

val_dataset = datasets.MNIST(
    "../mnist_data/", download=False, train=False, transform=transform
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True
)

num_batches = len(trn_loader)

data = next(iter(trn_loader))

img, label = data

# construct model on cuda if available

use_cuda = torch.cuda.is_available()


class CNNClassifier(nn.Module):
    def __init__(self):
        # 항상 torch.nn.Module을 상속받고 시작
        super(CNNClassifier, self).__init__()
        conv1 = nn.Conv2d(1, 6, 5, 1)  # 6@24*24
        # activation ReLU
        pool1 = nn.MaxPool2d(2)  # 6@12*12
        conv2 = nn.Conv2d(6, 16, 5, 1)  # 16@8*8
        # activation ReLU
        pool2 = nn.MaxPool2d(2)  # 16@4*4

        self.conv_module = nn.Sequential(
            conv1, nn.ReLU(), pool1, conv2, nn.ReLU(), pool2
        )

        fc1 = nn.Linear(16 * 4 * 4, 120)
        # activation ReLU
        fc2 = nn.Linear(120, 84)
        # activation ReLU
        fc3 = nn.Linear(84, 10)

        self.fc_module = nn.Sequential(fc1, nn.ReLU(), fc2, nn.ReLU(), fc3)

        # gpu로 할당
        if use_cuda:
            self.conv_module = self.conv_module.cuda()
            self.fc_module = self.fc_module.cuda()

    def forward(self, x):
        out = self.conv_module(x)  # @16*4*4
        # make linear
        dim = 1
        for d in out.size()[1:]:  # 16, 4, 4
            dim = dim * d
        out = out.view(-1, dim)
        out = self.fc_module(out)
        return F.softmax(out, dim=1)


cnn = CNNClassifier()

# loss
criterion = nn.CrossEntropyLoss()
# backpropagation method
learning_rate = 1e-3
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)

# hyper-parameters
num_epochs = 1

row_num = 2
col_num = 4

trn_loss_list = []
val_loss_list = []

print("학습하는데 약 3~4분 정도가 걸립니다...")

for epoch in range(num_epochs):
    trn_loss = 0.0
    for i, data in enumerate(trn_loader):
        x, label = data
        if use_cuda:
            x = x.cuda()
            label = label.cuda()
        # grad init
        optimizer.zero_grad()  # 초기화를 안하면 누적된다.
        # forward propagation
        model_output = cnn(x)
        # calculate loss
        loss = criterion(model_output, label)
        # back propagation
        loss.backward()
        # weight update
        optimizer.step()

        # trn_loss summary
        trn_loss += loss.item()
        # del (memory issue)
        del loss
        del model_output

        # 학습과정 출력
        if (i + 1) % 100 == 0:  # every 100 mini-batches
            with torch.no_grad():
                val_loss = 0.0
                for j, val in enumerate(val_loader):
                    val_x, val_label = val
                    if use_cuda:
                        val_x = val_x.cuda()
                        val_label = val_label.cuda()
                    val_output = cnn(val_x)
                    v_loss = criterion(val_output, val_label)
                    val_loss += v_loss

            # print("실제값: {}".format(val_label[:row_num*col_num]))
            # print("예측값: {}".format(val_output.argmax(dim=1)[:row_num*col_num]))
            del val_output
            del v_loss

            print("진행도: {}/{}".format(i + 1, num_batches))
            # print()

            trn_loss_list.append(trn_loss / 100)
            val_loss_list.append(val_loss / len(val_loader))
            trn_loss = 0.0

print("학습완료!")
print()
print("이제 입력받은 이미지를 가지고 실행해보겠습니다.")
image = Image.open(sys.argv[1])
image = transform(image)
image1 = torch.zeros((16, 1, 28, 28))
image1 += image

if use_cuda:
    image1 = image1.cuda()
# grad init
optimizer.zero_grad()
# forward propagation
model_output = cnn(image1)
print("예측값: {}".format(model_output.argmax(dim=1)[0]))
