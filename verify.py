from PIL import features
from torch.utils.data import DataLoader

import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import transforms

import matplotlib.pyplot as plt

from defParameter import *
from load_dataset import *
from VGG16Network import *
from train import *

custom_transform = transforms.Compose([transforms.Resize([224, 224]),transforms.ToTensor()])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.vgg16().to(device)

model.load_state_dict(torch.load("./NetworkModel/vgg16_best_model.pth",map_location="cuda:0"))

val_dataset = datasets.ImageFolder(
    root=r'./val',
    transform=custom_transform
)
classes = val_dataset.classes
val_loader = DataLoader(dataset=val_dataset,
                        batch_size=16,
                        shuffle=True)

for features, targets in val_loader:
    predictions = model.forward(features.to(device))
    predictions = torch.argmax(predictions, dim=1)
    plt.figure(figsize=(15, 15)) 

    for i in range(len(features)):
        plt.subplot(4 ,4, i+1)
        plt.title("Prediction:{}\nTarget:{}".format(classes[predictions[i]],classes[targets[i]]))
        img = features[i].swapaxes(0,1)
        img = img.swapaxes(1,2)
        plt.imshow(img)

        plt.axis('off')

    plt.show()
    break