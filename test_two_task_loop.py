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
    

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    for i, (inputs, labels) in enumerate(dataloaders['test']):
        inputs = inputs.to(device)

        outputs = model(inputs)
        outputs_main = outputs[:, :, 0]
        _, preds_main = torch.max(outputs_main, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds_main[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                model.train(mode=was_training)
                return
    model.train(mode=was_training)
    

def test(model, criterion, auxiliary_task, main_task=5):
    since = time.time()
    phase = 'test'
    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0
    one_one = 0
    one_zero = 0
    zero_one = 0
    zero_zero = 0

    # Iterate over data.
    for inputs, labels in dataloaders[phase]:
        main_labels = labels[:, main_task]
        aux_labels = labels[:, auxiliary_task]
        inputs = inputs.to(device)
        main_labels = main_labels.to(device)
        aux_labels = aux_labels.to(device)

        outputs = model(inputs)
        outputs_main = outputs[:, :, 0]
        outputs_aux = outputs[:, :, 1]
        _, preds_main = torch.max(outputs_main, 1)
        loss = criterion(outputs_main, main_labels) + criterion(outputs_aux, aux_labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds_main == main_labels.data)
        # needed for new metrics
        tot = main_labels.data + preds_main
        diff = main_labels.data - preds_main
        one_one += np.sum(np.asarray(tot == 2))
        zero_zero += np.sum(np.asarray(tot == 0))
        one_zero += np.sum(np.asarray(diff == 1))
        zero_one += np.sum(np.asarray(diff == -1))

    test_loss = running_loss / dataset_sizes[phase]
    test_acc = running_corrects.double() / dataset_sizes[phase]
    # new metrics
    recall_female = one_one / (one_one + one_zero)
    recall_male = zero_zero / (zero_zero + zero_one)
    precision_female = one_one / (one_one + zero_one)
    precision_male = zero_zero / (zero_zero + one_zero)
    f1_female = 2 * precision_female * recall_female / (precision_female + recall_female)
    f1_male = 2 * precision_male * recall_male / (precision_male + recall_male)

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Auxiliary task: {} - {} Loss: {:.4f} Acc: {:.4f}'.format(
        categories[auxiliary_task], phase, test_loss, test_acc))
    print('FEMALE: Recall: {}, Precision: {}, F1: {}'.format(recall_female, precision_female, f1_female))
    print('MALE: Recall: {}, Precision: {}, F1: {}'.format(recall_male, precision_male, f1_male))
    print()


# List of categories
categories = ['black', 'blue', 'brown', 'collar', 'cyan', 'gender', 'gray', 'green',
              'many_colors', 'necktie', 'pattern_floral', 'pattern_graphics', 'pattern_plaid',
              'pattern_solid', 'pattern_spot', 'pattern_stripe', 'placket', 'purple', 'red', 'scarf',
              'skin_exposure', 'white', 'yellow']

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

main_task = 5  # gender
class_names = ['Male', 'Female']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


for auxiliary_task in range(len(categories)):
    if auxiliary_task != main_task:

        # Load the data
        image_datasets = {x: ClothingAttributeDataset(labels_dir, images_dir, x, data_transforms[x]) for x in ['test']}
        dataloaders = {x: DataLoader(image_datasets[x], batch_size=TEST_BATCH_SIZE) for x in ['test']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}

        # Load the saved model
        model = torch.load('./exp4/two_task_aux{}.pt'.format(auxiliary_task))

        # GPU or CPU
        model = model.to(device)

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Print out the test results (run only once)
        test(model, criterion, auxiliary_task)

        # Some figures from the test set
        #visualize_model(model)
        #plt.ioff()
        #plt.show()
