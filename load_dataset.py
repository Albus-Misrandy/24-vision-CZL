import torchvision.datasets
import numpy as np
from torchvision import datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision.transforms import transforms

data_transform = transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor()])

Batch_size = 16

train_dataset =datasets.ImageFolder(root='./train',transform = data_transform)
train_dataloader = DataLoader(dataset=train_dataset,batch_size=Batch_size,shuffle=True,num_workers=2)
test_dataset = datasets.ImageFolder(root='./test',transform=data_transform)
test_dataloader = DataLoader(dataset=test_dataset,batch_size=Batch_size,shuffle=True,num_workers=2)

classes = test_dataset.classes

cnf_matrix = np.zeros([len(classes),len(classes)])