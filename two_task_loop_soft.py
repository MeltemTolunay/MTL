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

# List of categories
categories = ['black', 'blue', 'brown', 'collar', 'cyan', 'gender', 'gray', 'green',
              'many_colors', 'necktie', 'pattern_floral', 'pattern_graphics', 'pattern_plaid',
              'pattern_solid', 'pattern_spot', 'pattern_stripe', 'placket', 'purple', 'red', 'scarf',
              'skin_exposure', 'white', 'yellow']


# Function to display a single tensor as an image
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


# Define the main task
main_task = 5  # gender
class_names = ['Male', 'Female']

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

# Open file to record accuracies for each task
f = open('best_accuracies_{}.txt'.format(main_task), 'a+')  # CHANGE THIS NAME EVERY TIME YOU RUN IT!!!!!!!!!

# Check GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model1, model2, criterion, optimizer, scheduler, auxiliary_task, num_epochs=25):
    since = time.time()

    reg_layers = [0, 3, 6, 9, 12, 15, 18, 24, 27, 30, 33, 39, 42, 45, 48, 54, 57]

    best_model_wts1 = copy.deepcopy(model1.state_dict())
    best_model_wts2 = copy.deepcopy(model2.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Auxiliary task: {} - Epoch {}/{}'.format(categories[auxiliary_task], epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model1.train()  # Set model to training mode
                model2.train()
            else:
                model1.eval()   # Set model to evaluate mode
                model2.eval()

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
                    outputs_main = model1(inputs)
                    outputs_aux = model2(inputs)
                    _, preds_main = torch.max(outputs_main, 1)
                    _, preds_aux = torch.max(outputs_aux, 1)
                    # For now, add losses with the same weight=[1,1]
                    l2_reg = None
                    for layer in reg_layers:
                        W1 = list(model1.parameters())[layer]
                        W2 = list(model2.parameters())[layer]
                        if l2_reg is None:
                            l2_reg = torch.sum((W1 - W2) ** 2)
                            # zero = torch.zeros_like(W1)
                            # l2_reg = criterion_reg(W1 - W2, zero)
                        else:
                            l2_reg += torch.sum((W1 - W2) ** 2)
                            # zero = torch.zeros_like(W1)
                            # l2_reg += criterion_reg(W1 - W2, zero)

                    loss = criterion(outputs_main, main_labels) + \
                           ALPHA * criterion(outputs_aux, aux_labels) + \
                           1 * l2_reg

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
                best_model_wts1 = copy.deepcopy(model1.state_dict())
                best_model_wts2 = copy.deepcopy(model2.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    f.write('{}: {}\n'.format(categories[auxiliary_task], best_acc))
    print()

    # load best model weights
    model1.load_state_dict(best_model_wts1)
    model2.load_state_dict(best_model_wts2)

    return model1


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)

            outputs_main = model(inputs)
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


for aux in range(23):
    if aux != main_task:
        auxiliary_task = aux  # looped over

        # Load the data
        image_datasets = {x: ClothingAttributeDataset(labels_dir, images_dir, x, data_transforms[x])
                          for x in ['train', 'val']}
        dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True)
                       for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x])
                         for x in ['train', 'val']}

        loss_history = []
        accuracy_history = []

        # Finetune the convnet
        model_conv1 = models.resnet18(pretrained=True)
        num_ftrs1 = model_conv1.fc.in_features
        model_conv1.fc = nn.Linear(num_ftrs1, 2)
        model_conv1 = model_conv1.to(device)

        model_conv2 = models.resnet18(pretrained=True)
        num_ftrs2 = model_conv2.fc.in_features
        model_conv2.fc = nn.Linear(num_ftrs2, 2)
        model_conv2 = model_conv2.to(device)

        criterion = nn.CrossEntropyLoss()

        # params = list(model.fc1.parameters()) + list(model.fc2.parameters())
        # Observe that all parameters are being optimized as opposed to above.
        params = list(model_conv1.parameters()) + list(model_conv2.parameters())
        optimizer_conv = optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=STEP_SIZE, gamma=GAMMA)

        model = train_model(model_conv1, model_conv2, criterion, optimizer_conv, exp_lr_scheduler, aux, num_epochs=25)
        torch.save(model, 'two_task_soft_aux{}.pt'.format(aux))

        #np.savetxt('loss_history_aux{}.txt'.format(aux), loss_history)
        #np.savetxt('acc_aux{}.txt'.format(aux), accuracy_history)

f.close()
