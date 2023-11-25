'''Train CIFAR10 with PyTorch.'''
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms, datasets

import os
import argparse

from models import *
# from utils import progress_bar

print(os.getcwd())
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

best_epoch = 0
best_loss = 0
keep_train_loss = [] # 记录每次训练的loss
keep_test_loss = [] # 记录每次测试的loss
keep_train_acc = [] # 记录每次训练的正确率
keep_test_acc = [] # 记录每次测试的正确率

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
net = SimpleDLA()
net = net.to(device)
net_name = 'DLA'
if device == 'cuda':
    from torch.backends.cudnn import cudnn
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+net_name+'.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                      momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4) #added
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print('Loss: %.3f | Acc: %.3f%% (%d/%d)'% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    else:
        keep_train_loss.append(train_loss/(batch_idx+1)) #add
        keep_train_acc.append(100.*correct/total) #add
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    global best_loss #added
    global best_epoch #added
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (test_loss/(batch_idx+1), 100.*correct/total, correct, total)) #modified
        else: #added
            keep_test_loss.append(test_loss/(batch_idx+1)) #added
            keep_test_acc.append(100*correct/total) #added

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        best_epoch = epoch #add
        best_acc = acc #add
        best_loss = test_loss / (batch_idx + 1) #add
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/{net_name}.pth')

start_time = time() #added

keep_epoch = range(start_epoch, start_epoch+50) # 每次训练的epoch
for epoch in keep_epoch:
    train(epoch)
    test(epoch)
    scheduler.step()

end_time = time() #added

def save_data():
    if not os.path.isdir('fig'):
        os.mkdir('fig')
    with open(f'./fig/{net_name}_acc.csv', 'w') as f:
        for i in len(keep_epoch):
            f.write('%d,%f,%f\n'
                    % (keep_epoch[i], keep_test_acc[i], keep_train_acc[i]))
    with open(f'./fig/{net_name}_loss.csv', 'w') as f:
        for i in len(keep_epoch):
            f.write('%d,%f,%f\n'
                    % (keep_epoch[i], keep_test_loss[i], keep_train_loss[i]))

print('runtime : %.3f'%((end_time - start_time)/60)) #add
print('Test: Loss: %.3f | Best_Acc: %.3f | epoch: %d' % (best_loss, best_acc, best_epoch)) #add
print('Train: Loss: %.3f | Acc: %.3f' % (keep_train_loss[best_epoch], keep_train_acc[best_epoch]))
save_data()
