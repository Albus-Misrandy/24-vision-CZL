from torch.optim import lr_scheduler
import numpy as np
from load_dataset import *
from VGG16Network import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = MyNetwork(num_classes=len(classes)).to(device)

loss = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)

epoch = 5

train_acc_lst,test_acc_lst = [],[]
train_loss_lst,test_loss_lst = [],[]

max_train_acc = 0
max_test_acc = 0

def transfer_channel(image):
    image = np.array(image)
    image = image.transpose((1,0,2,3))
    image = np.concatenate((image,image,image),axis=0)
    image = image.transpose((1,0,2,3))
    image = torch.tensor(image)
    return image

def computer_accuracy_and_loss(model,dataset,data_loader,device):
    correct , total = .0,.0
    for i, (features,targets) in enumerate(data_loader):
        if features.size(1)==1:
            features = transfer_channel(features)
        features = features.to(device)
        targets = targets.to(device)
        output = model(features)
        currnet_loss = loss(output,targets)

        _, predicted_labels = torch.max(output, 1)
        correct += (predicted_labels == targets).sum()

        for idx in range(len(targets)):
            cnf_matrix[targets[idx]][predicted_labels[idx]] += 1
        total += targets.size(0)

    return float(correct)*100/len(dataset),currnet_loss.item()