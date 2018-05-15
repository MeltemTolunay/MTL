import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import time
from custom_dataset import ClothingAttributeDataset
from config import *


class TwoTaskModel(nn.Module):
    def __init__(self, feature_size, num_classes1, num_classes2, pretrained_model):
        super().__init__()
        # assign layer objects to class attributes
        self.fc1 = nn.Linear(feature_size, num_classes1)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(feature_size, num_classes2)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.pretrained_model = pretrained_model
        self.body = nn.Sequential(*list(self.pretrained_model.children())[:-1])

    def forward(self, x):
        # forward always defines connectivity
        x = self.body(x)
        x = x.view(x.shape[0], x.shape[1])
        score_main = torch.unsqueeze(self.fc1(x), 2)
        score_aux = torch.unsqueeze(self.fc2(x), 2)
        scores = torch.cat((score_main, score_aux), 2)  # (N,C,T)  C:{0,1}  T:number of tasks (2)
        # print(scores.shape)  # torch.Size([16, 2, 2])
        return scores


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'test': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
}

# Image and label directories
labels_dir = './ClothingAttributeDataset/labels/'
images_dir = './ClothingAttributeDataset/images/'

# Load the data
image_datasets = {x: ClothingAttributeDataset(labels_dir, images_dir, x, data_transforms[x]) for x in ['test']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=TEST_BATCH_SIZE) for x in ['test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}

main_task = 5  # gender
auxiliary_task = 20  # skin exposure
class_names = ['Male', 'Female']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


'''
# Visualize the data
images, labels = next(iter(dataloaders['train']))
print(images.shape)
print(labels.shape)

attribute_num = 5  # gender
title = ['male' if labels[x][attribute_num] == torch.tensor([0]) else 'female' for x in range(BATCH_SIZE)]
out = torchvision.utils.make_grid(images)
imshow(out, title=title)
plt.ioff()  # otherwise the plot will weirdly disappear
plt.show()
'''


def test(model1, model2, criterion):
    since = time.time()
    phase = 'test'
    model1.eval()  # Set model to evaluate mode
    model2.eval()  # Set model to evaluate mode

    running_loss1 = 0.0
    running_corrects1 = 0

    running_loss2 = 0.0
    running_corrects2 = 0

    match1 = []
    match2 = []

    # Iterate over data.
    for inputs, labels in dataloaders[phase]:
        main_labels = labels[:, main_task]
        aux_labels = labels[:, auxiliary_task]
        inputs = inputs.to(device)
        main_labels = main_labels.to(device)
        aux_labels = aux_labels.to(device)

        outputs1 = model1(inputs)
        _, preds_main1 = torch.max(outputs1, 1)
        loss1 = criterion(outputs1, main_labels)

        outputs2 = model2(inputs)
        outputs_main2 = outputs2[:, :, 0]
        outputs_aux2 = outputs2[:, :, 1]
        _, preds_main2 = torch.max(outputs_main2, 1)
        loss2 = criterion(outputs_main2, main_labels) + criterion(outputs_aux2, aux_labels)

        # statistics
        running_loss1 += loss1.item() * inputs.size(0)
        running_corrects1 += torch.sum(preds_main1 == main_labels.data)
        match1.append(preds_main1 == main_labels.data)

        running_loss2 += loss2.item() * inputs.size(0)
        running_corrects2 += torch.sum(preds_main2 == main_labels.data)
        match2.append(preds_main2 == main_labels.data)

    test_loss1 = running_loss1 / dataset_sizes[phase]
    test_acc1 = running_corrects1.double() / dataset_sizes[phase]

    test_loss2 = running_loss2 / dataset_sizes[phase]
    test_acc2 = running_corrects2.double() / dataset_sizes[phase]

    print(match1)
    print(match2)
    diff = match2[0] - match1[0]
    print(diff)  # 1 where two-model was correct but single-model was wrong
    n = 1485
    picture = []
    for d in diff:
        if d == 1:
            picture.append(n)
        n += 1
    print(picture)

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Model 1: {} Loss: {:.4f} Acc: {:.4f}'.format(
        phase, test_loss1, test_acc1))
    print('Model 2: {} Loss: {:.4f} Acc: {:.4f}'.format(
        phase, test_loss2, test_acc2))


def visualize_model1(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    for i, (inputs, labels) in enumerate(dataloaders['test']):
        inputs = inputs.to(device)
        main_labels = labels[:, main_task]
        main_labels = main_labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}/{}'.format(class_names[preds[j]], class_names[main_labels.data[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                model.train(mode=was_training)
                return
    model.train(mode=was_training)


def visualize_model2(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    for i, (inputs, labels) in enumerate(dataloaders['test']):
        inputs = inputs.to(device)
        main_labels = labels[:, main_task]
        main_labels = main_labels.to(device)

        outputs = model(inputs)
        outputs_main = outputs[:, :, 0]
        _, preds_main = torch.max(outputs_main, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}/{}'.format(class_names[preds_main[j]], class_names[main_labels.data[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                model.train(mode=was_training)
                return
    model.train(mode=was_training)


# Load the saved models
model1 = torch.load('baseline_single_task.pt')
model2 = torch.load('baseline_two_task.pt')

# GPU or CPU
model1 = model1.to(device)
model2 = model2.to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Print out the test results (run only once)
test(model1, model2, criterion)

# Some figures from the test set
visualize_model1(model1, num_images=10)
visualize_model2(model2, num_images=10)
plt.ioff()
plt.show()

'''
/Users/Meltem/anaconda3/envs/pytorch04/bin/python /Users/Meltem/PycharmProjects/MTL/compare.py
[tensor([ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  0,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
         0,  0,  0,  1,  1,  1,  0,  1,  0,  1,  1,  1,  1,  0,
         0,  1,  1,  0,  0,  1,  0,  1,  0,  1,  0,  0,  0,  0,
         0,  1,  0,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,
         1,  1,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  0,  1,
         0,  1,  1,  1,  1,  0,  1,  1,  0,  0,  1,  1,  0,  1,
         0,  0,  0,  0,  1,  1,  0,  0,  1,  0,  1,  1,  0,  1,
         0,  1,  1,  1,  0,  1,  1,  1,  0,  1,  0,  0,  0,  0,
         1,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  1,  0,
         1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,
         1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,
         1,  1,  0,  0,  1,  0,  1,  0,  0,  1,  0,  0,  1,  1,
         1,  1,  1,  1,  1,  0,  1,  0,  1,  1,  1,  1,  1,  1,
         1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,
         0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  0,
         1,  1,  1,  1,  0,  1,  1,  1,  0,  1,  1,  1,  0,  1,
         0,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,
         1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  0,  1,
         1,  1,  1,  1,  1,  1,  1,  1], dtype=torch.uint8)]
[tensor([ 0,  0,  1,  1,  0,  1,  0,  1,  1,  1,  1,  1,  1,  1,
         1,  0,  0,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  0,
         1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  0,  0,  1,
         1,  0,  1,  0,  1,  1,  1,  1,  1,  0,  0,  1,  0,  1,
         1,  1,  0,  1,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,
         0,  0,  0,  1,  0,  1,  1,  1,  0,  1,  1,  1,  0,  0,
         0,  0,  1,  1,  0,  0,  1,  1,  0,  1,  0,  1,  0,  1,
         0,  0,  1,  1,  1,  1,  0,  1,  0,  0,  1,  1,  0,  0,
         1,  1,  1,  1,  0,  0,  1,  1,  0,  0,  1,  0,  1,  0,
         1,  0,  1,  0,  1,  1,  1,  1,  1,  0,  0,  0,  1,  0,
         0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
         0,  0,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  0,  1,
         1,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
         0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,
         0,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
         1,  0,  1,  0,  1,  1,  1,  1,  0,  1,  1,  0,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1], dtype=torch.uint8)]
tensor([   0,  255,    0,    0,  255,    0,  255,    0,    0,    0,
           0,    0,    0,    0,    0,  255,  255,    0,  255,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    1,    0,  255,
           0,    0,    0,    0,    0,    0,  255,    0,    0,    0,
           1,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    1,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    1,    1,    1,    0,
           0,    0,    1,    0,    0,    0,    0,  255,  255,    1,
           1,  255,    0,    0,    1,    0,    1,    0,    1,  255,
           0,    1,    0,    1,    1,    0,    0,    0,    0,    0,
           0,    0,    0,  255,    0,    0,    0,    0,  255,  255,
         255,    0,    0,    0,    0,    1,  255,    0,    1,    0,
           0,  255,    0,  255,    0,    0,  255,    0,    0,    0,
           0,    1,  255,    0,    0,    0,    0,    0,    1,    1,
           0,    0,    0,    1,  255,    0,    0,    0,    0,  255,
           1,    0,    0,    0,    0,  255,    0,    0,    0,  255,
           1,    0,    1,    0,    0,  255,    0,    0,    0,    0,
           1,    0,    0,    0,  255,  255,    0,    0,  255,    0,
           1,    0,    0,    0,    0,    0,    1,    0,    0,    0,
           0,    0,    0,    0,    1,    0,    0,    0,    0,    0,
           1,    0,    0,    0,    0,    0,  255,  255,    1,    1,
         255,    0,    0,    1,    1,    0,    0,    0,  255,    0,
           0,    0,    0,    0,    0,    0,  255,    1,    0,    0,
           0,    0,    0,    0,    0,    1,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    1,    1,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    1,    0,
           0,    1,    0,    0,    0,    0,    0,    0,    0,    0,
           1,    0,    0,    0,    1,    0,    0,    0,  255,    0,
           0,    0,    0,    0,    1,    0,    0,    0,    0,    0,
           0,  255,    0,    0,    0,    0,    0,    0,    0,    1,
           0,  255,    1,    0,    0,    0,    0,    0,    0,    0,
           0,    0], dtype=torch.uint8)
           
[1552, 1565, 1600, 1611, 1612, 1613, 1617, 1624, 1625, 1629, 1631, 1633, 1636, 1638, 1639, 1660, 1663, 1676, 1683, 1684,
 1688, 1695, 1705, 1707, 1715, 1725, 1731, 1739, 1745, 1753, 1754, 1758, 1759, 1772, 1780, 1791, 1792, 1803, 1806, 1815,
 1819, 1829, 1844, 1847]
 
Testing complete in 1m 37s
Model 1: test Loss: 0.4475 Acc: 0.7823
Model 2: test Loss: 0.9454 Acc: 0.8011

'''
