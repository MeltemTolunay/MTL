import torch
from torch.utils.data import Dataset, DataLoader, sampler
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import scipy.io
import os
import time
import copy


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


def visualize_model(model, dataloaders, num_images=6, use_gpu=False, titles=None):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dataloaders['val']):
        inputs, labels = data

        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            if titles is not None:
                # title = class_names[preds[j]]
                title = titles
                ax.set_title('Predicted: {}'.format(title))
            imshow(inputs.cpu().data[j])

            if num_images == images_so_far:
                model.train(mode=was_training)
                return

    model.train(mode=was_training)


class ClothingAttributeDataset(Dataset):
    """Clothing attributes dataset."""

    def __init__(self, labels_dir, images_dir, transform=None, train=True):
        """
        :param labels_dir: Path to the labels folder.
        :param images_dir: Path to the images folder.
        :param transform: Optional transform. (But make sure images are the same size if you choose to omit!)
        :param train: Train mode or test mode
        """
        # Train mode or test mode
        self.train = train

        # Labeling categories
        self.categories = ['black', 'blue', 'brown', 'category', 'collar', 'cyan', 'gender', 'gray', 'green',
                           'many_colors', 'neckline', 'necktie', 'pattern_floral', 'pattern_graphics', 'pattern_plaid',
                           'pattern_solid', 'pattern_spot', 'pattern_stripe', 'placket', 'purple', 'red', 'scarf',
                           'skin_exposure', 'sleevelength', 'white', 'yellow']

        # Ground truths for each category as a column vector
        self.attributes = {}
        directory = os.fsencode(labels_dir)
        for i, file in enumerate(os.listdir(directory)):
            dir = os.path.join(directory, file)
            self.attributes[self.categories[i]] = scipy.io.loadmat(dir)['GT']  # (1856, 1)

        # Ground truth matrix (1856 x 26)
        self.attributes_matrix = self.attributes['black']
        for category in self.categories[1:]:
            self.attributes_matrix = np.hstack((self.attributes_matrix, self.attributes[category]))
            self.attributes_matrix[(self.attributes_matrix == float('nan'))] == 1

        self.attributes_matrix += -1  # Change labels from (1,2) to (0,1)
        self.attributes_matrix = self.attributes_matrix.astype(int)

        if self.train:
            self.attributes_matrix = self.attributes_matrix[:1484]
        else:
            self.attributes_matrix = self.attributes_matrix[1484:]

        # Images directory
        self.images_dir = images_dir

        # List of image names as strings
        self.images_list = []
        directory = os.fsencode(images_dir)
        for i, file in enumerate(os.listdir(directory)):
            filename = file.decode('utf-8')
            self.images_list.append(filename)
        if self.train:
            self.images_list = self.images_list[:1484]
        else:
            self.images_list = self.images_list[1484:]

        # Transforms
        self.transform = transform

    def __len__(self):
        return len(self.images_list)  # 1484(train) or 372(test)

    def __getitem__(self, index):

        # Full data path to the image
        image_name = os.path.join(self.images_dir, self.images_list[index])

        # Image in MxNxC
        image = io.imread(image_name)

        # Transforms (at least call ToTensor here to transpose to CxMxN)
        if self.transform:
            image = self.transform(image)
        else:
            self.to_tensor = transforms.ToTensor()
            image = self.to_tensor(image)

        # All labels by taking the row
        labels = np.asarray(self.attributes_matrix[index, :])

        # Dictionary to be returned
        sample = {'image': image, 'labels': labels}
        #sample['image'].shape ==> torch.Size([3, 256, 256])
        #sample['labels'].shape ==> (26,)

        return sample


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, task, num_epochs=2, use_gpu=False):
    tic = time.time()

    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # EPOCHS
    for epoch in range(num_epochs):
        print('Epoch: {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)

        # EACH EPOCH HAS TRAIN AND VAL
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()  # called once every epoch
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            # ITERATE OVER DATA
            for data in dataloaders[phase]:
                inputs = data['image']
                labels = data['labels']
                labels = labels[:, task]

                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels)

                optimizer.zero_grad() # called every iteration for a new batch

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_accuracy = float(running_corrects) / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_accuracy))

            if phase == 'val' and epoch_accuracy > best_acc:
                best_acc = epoch_accuracy
                best_model_weights = copy.deepcopy(model.state_dict())

        print()

    toc = time.time()
    time_elapsed = toc - tic
    print('Training complete in {:.0f} mins {:.0f} secs'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val accuracy: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_weights)
    return model


def main():
    use_gpu = torch.cuda.is_available()

    labels_dir = 'ClothingAttributeDataset/labels/'
    images_dir = 'ClothingAttributeDataset/images/'

    # Data augmentation and transforms for TRAINING
    data_transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # Correct these values later?
    ])

    data_transform_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # Correct these values later?
    ])

    data_transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # Correct these values later?
    ])

    dataset_train = ClothingAttributeDataset(labels_dir, images_dir, data_transform_train, train=True)
    dataset_test = ClothingAttributeDataset(labels_dir, images_dir, data_transform_test, train=False)

    # Now random splitting for train-val
    num_train = len(dataset_train)
    indices = list(range(num_train))
    split = 372

    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))
    train_sampler = sampler.SubsetRandomSampler(train_idx)
    validation_sampler = sampler.SubsetRandomSampler(validation_idx)

    train_loader = DataLoader(dataset_train, batch_size=4, sampler=train_sampler)  # shuffle=True ?
    val_loader = DataLoader(dataset_train, batch_size=4, sampler=validation_sampler)  # shuffle=True ?
    test_loader = DataLoader(dataset_test, batch_size=4)  # shuffle=True ?

    sample = next(iter(train_loader))
    images = sample['image']
    labels = sample['labels']
    print(images.shape)
    print(labels.shape)

    # Uncomment to visualize the data
    #out = torchvision.utils.make_grid(images)
    #imshow(out)
    #plt.ioff()  # otherwise the plot will weirdly disappear
    #plt.show()

    dataloaders = {}
    dataloaders['train'] = train_loader
    dataloaders['val'] = val_loader

    dataset_sizes = {}
    dataset_sizes['train'] = 1484
    dataset_sizes['val'] = 372

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)

    if use_gpu:
        model_ft = model_ft.cuda()

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, task=11)

    visualize_model(model_ft, dataloaders, num_images=6)

    plt.ioff()  # otherwise the plot will weirdly disappear
    plt.show()

if __name__ == "__main__":
    main()








