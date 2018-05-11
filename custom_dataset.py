from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from skimage import io
import scipy.io
import os


class ClothingAttributeDataset(Dataset):
    """Clothing attributes dataset."""

    def __init__(self, labels_dir, images_dir, mode='train', transform=None):
        """
        :param labels_dir: Path to the labels folder.
        :param images_dir: Path to the images folder.
        :param mode: 'train' or 'val' or 'test'
        :param transform: Optional transform. (But make sure images are the same size if you choose to omit!)
        """
        # Train or validation or test
        self.mode = mode

        # Labeling categories
        self.categories = ['black', 'blue', 'brown', 'collar', 'cyan', 'gender', 'gray', 'green',
                           'many_colors', 'necktie', 'pattern_floral', 'pattern_graphics', 'pattern_plaid',
                           'pattern_solid', 'pattern_spot', 'pattern_stripe', 'placket', 'purple', 'red', 'scarf',
                           'skin_exposure', 'white', 'yellow']

        # Ground truths for each category as a column vector
        self.attributes = {}
        directory = os.fsencode(labels_dir)
        for i, file in enumerate(os.listdir(directory)):
            dir = os.path.join(directory, file)
            self.attributes[self.categories[i]] = scipy.io.loadmat(dir)['GT']  # (1856, 1)

        # Ground truth matrix (1856 x 23)
        self.attributes_matrix = self.attributes['black']
        for category in self.categories[1:]:
            self.attributes_matrix = np.hstack((self.attributes_matrix, self.attributes[category]))

        self.attributes_matrix += -1  # Change labels from (1,2) to (0,1)
        self.attributes_matrix[np.isnan(self.attributes_matrix)] = 0
        self.attributes_matrix = self.attributes_matrix.astype(int)

        # Train-val-test separation for labels
        if self.mode == 'train':
            self.attributes_matrix = self.attributes_matrix[:1296]
        elif self.mode == 'val':
            self.attributes_matrix = self.attributes_matrix[1296:1484]
        elif self.mode == 'test':
            self.attributes_matrix = self.attributes_matrix[1484:]
        else:
            raise ValueError('Mode must be train, val or test.')

        # Images directory
        self.images_dir = images_dir

        # List of image names as strings
        self.images_list = []
        directory = os.fsencode(images_dir)
        for i, file in enumerate(os.listdir(directory)):
            filename = file.decode('utf-8')
            self.images_list.append(filename)

        # Train-val-test separation for images
        if self.mode == 'train':
            self.images_list = self.images_list[:1296]
        elif self.mode == 'val':
            self.images_list = self.images_list[1296:1484]
        elif self.mode == 'test':
            self.images_list = self.images_list[1484:]
        else:
            raise ValueError('Mode must be train, val or test.')

        # Transforms
        self.transform = transform

    def __len__(self):
        return len(self.images_list)  # 1112(train) or 372(val) or 372(test)

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

        # Pair to be returned
        #image.shape ==> torch.Size([3, 256, 256])
        #labels.shape ==> (26,)

        return image, labels