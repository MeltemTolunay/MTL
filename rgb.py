import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models
import numpy as np
import time
import copy
from custom_dataset import ClothingAttributeDataset
from config import *

# List of categories
categories = ['black', 'blue', 'brown', 'collar', 'cyan', 'gender', 'gray', 'green',
              'many_colors', 'necktie', 'pattern_floral', 'pattern_graphics', 'pattern_plaid',
              'pattern_solid', 'pattern_spot', 'pattern_stripe', 'placket', 'purple', 'red', 'scarf',
              'skin_exposure', 'white', 'yellow']

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
aux_task1 = 1  # blue
aux_task2 = 7  # green
aux_task3 = 18  # red

class_names = ['Male', 'Female']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
                aux1_labels = labels[:, aux_task1]
                aux2_labels = labels[:, aux_task2]
                aux3_labels = labels[:, aux_task3]

                inputs = inputs.to(device)
                main_labels = main_labels.to(device)
                aux1_labels = aux1_labels.to(device)
                aux2_labels = aux2_labels.to(device)
                aux3_labels = aux3_labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs_main = outputs[:, :, 0]
                    outputs_aux1 = outputs[:, :, 1]
                    outputs_aux2 = outputs[:, :, 2]
                    outputs_aux3 = outputs[:, :, 3]
                    _, preds_main = torch.max(outputs_main, 1)
                    _, preds_aux1 = torch.max(outputs_aux1, 1)
                    _, preds_aux2 = torch.max(outputs_aux2, 1)
                    _, preds_aux3 = torch.max(outputs_aux3, 1)
                    # For now, add losses with the same weight=[1,1]
                    loss = criterion(outputs_main, main_labels) + \
                           ALPHA * criterion(outputs_aux1, aux1_labels) + \
                           ALPHA * criterion(outputs_aux2, aux2_labels) + \
                           ALPHA * criterion(outputs_aux3, aux3_labels)

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


class MultiTaskModel(nn.Module):
    def __init__(self, feature_size, num_classes, pretrained_model):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.body = nn.Sequential(*list(self.pretrained_model.children())[:-1])
        # assign layer objects to class attributes
        self.fc1 = nn.Linear(feature_size, num_classes)  # main
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(feature_size, num_classes)  # aux1
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(feature_size, num_classes)  # aux2
        nn.init.xavier_uniform_(self.fc3.weight)
        self.fc4 = nn.Linear(feature_size, num_classes)  # aux3
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x):
        # forward always defines connectivity
        x = self.body(x)
        x = x.view(x.shape[0], x.shape[1])
        score_main = torch.unsqueeze(self.fc1(x), 2)
        score_aux1 = torch.unsqueeze(self.fc2(x), 2)
        score_aux2 = torch.unsqueeze(self.fc3(x), 2)
        score_aux3 = torch.unsqueeze(self.fc4(x), 2)
        scores = torch.cat((score_main, score_aux1, score_aux2, score_aux3), 2)  # (N,C,T)  C:{0,1}  T: # of tasks (4)
        # print(scores.shape)  # torch.Size([16, 2, 4])
        return scores


# Finetune the convnet
model_conv = models.resnet18(pretrained=True)
num_ftrs = model_conv.fc.in_features
model = MultiTaskModel(num_ftrs, 2, model_conv).to(device)

criterion = nn.CrossEntropyLoss()

# Observe all parameters are being optimized.
params = model.parameters()
optimizer_conv = optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=STEP_SIZE, gamma=GAMMA)

model = train_model(model, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25)
torch.save(model, 'rgb.pt')

np.savetxt('loss_history_rgb.txt', loss_history)
np.savetxt('acc_rgb.txt', accuracy_history)

