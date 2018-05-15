import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from custom_dataset import ClothingAttributeDataset
from config import *


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
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
image_datasets = {x: ClothingAttributeDataset(labels_dir, images_dir, x, data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

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

loss_history = []
accuracy_history = []


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                main_labels = labels[:, main_task]
                aux_labels = labels[:, auxiliary_task]
                inputs = inputs.to(device)
                main_labels = main_labels.to(device)
                aux_labels = aux_labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs_main = outputs[:, :, 0]
                    outputs_aux = outputs[:, :, 1]
                    _, preds_main = torch.max(outputs_main, 1)
                    _, preds_aux = torch.max(outputs_aux, 1)
                    # For now, add losses with the same weight=[1,1]
                    loss = criterion(outputs_main, main_labels) + ALPHA * criterion(outputs_aux, aux_labels)  

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                loss_history.append(loss.item())
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds_main == main_labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if phase == 'val':
                accuracy_history.append(epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)

            outputs = model(inputs)
            outputs_main = outputs[:, :, 0]
            _, preds = torch.max(outputs_main, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


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


# Finetune the convnet
model_conv = models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

num_ftrs = model_conv.fc.in_features
model = TwoTaskModel(num_ftrs, 2, 2, model_conv)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as opposed to before.
params = list(model.fc1.parameters()) + list(model.fc2.parameters())
optimizer_conv = optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=STEP_SIZE, gamma=GAMMA)

model = train_model(model, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=NUM_EPOCHS)
torch.save(model, 'baseline_two_task.pt')

np.savetxt('loss_history_two.txt', loss_history)
np.savetxt('acc_two.txt', accuracy_history)

visualize_model(model)

plt.ioff()
plt.show()
