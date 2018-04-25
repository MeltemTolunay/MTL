from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from skimage import io
import scipy.io
import os


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


def main():
    labels_dir = 'ClothingAttributeDataset/labels/'
    images_dir = 'ClothingAttributeDataset/images/'

    # Data augmentation and transforms for TRAINING
    data_transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # Correct these values later?
    ])
    
    # Data augmentation and transforms for VALIDATION
    data_transform_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # Correct these values later?
    ])
       
    # Data augmentation and transforms for TESTING
    data_transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # Correct these values later?
    ])

    dataset_train = ClothingAttributeDataset(labels_dir, images_dir, data_transform_train, train=True)
    dataset_test = ClothingAttributeDataset(labels_dir, images_dir, data_transform_test, train=False)
    # Not sure how to split for val set

    print(len(dataset_train))
    print(len(dataset_test))

    train_loader = DataLoader(dataset_train, batch_size=4, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=4, shuffle=True)

    sample = next(iter(train_loader))
    images = sample['image']
    labels = sample['labels']
    print(images.shape)
    print(labels.shape)

    sample_test = next(iter(test_loader))
    images_test = sample_test['image']
    labels_test = sample_test['labels']
    print(images_test.shape)
    print(labels_test.shape)


if __name__ == "__main__":
    main()



'''
TODO:
- Compute the correct mean and std
- Data augmentation same as in A. Krizhevsky et al.
- Divide into train, test, val
'''
