from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from skimage import io
import scipy.io
import os

# download http://cs.brown.edu/~gmpatter/Attributes/SUNAttributeDB.tar.gz
# unzip, you will have a directory named SUNAtrributeDB
# download http://cs.brown.edu/~gmpatter/Attributes/SUNAttributeDB_Images.tar.gz
# unzip, move images directory into SUNAttributeDB
# In altogether.py change:
# from sun_dataset import SUNDataset
# labels_dir = './SUNAttributeDB/'
# images_dir = './SUNAttributeDB/images/'
# image_datasets = {x: SUNDataset(labels_dir, images_dir, x, data_transforms[x]) for x in ['train', 'val']}
# choose main task from list below, change main task index accordingly
'''
attributes = 

    'sailing/ boating'
    'driving'
    'biking'
    'transporting things or people'
    'sunbathing'
    'vacationing/ touring'
    'hiking'
    'climbing'
    'camping'
    'reading'
    'studying/ learning'
    'teaching/ training'
    'research'
    'diving'
    'swimming'
    'bathing'
    'eating'
    'cleaning'
    'socializing'
    'congregating'
    'waiting in line/ queuing'
    'competing'
    'sports'
    'exercise'
    'playing'
    'gaming'
    'spectating/ being in an audience'
    'farming'
    'constructing/ building'
    'shopping'
    'medical activity'
    'working'
    'using tools'
    'digging'
    'conducting business'
    'praying'
    'fencing'
    'railing'
    'wire'
    'railroad'
    'trees'
    'grass'
    'vegetation'
    'shrubbery'
    'foliage'
    'leaves'
    'flowers'
    'asphalt'
    'pavement'
    'shingles'
    'carpet'
    'brick'
    'tiles'
    'concrete'
    'metal'
    'paper'
    'wood (not part of a tree)'
    'vinyl/ linoleum'
    'rubber/ plastic'
    'cloth'
    'sand'
    'rock/stone'
    'dirt/soil'
    'marble'
    'glass'
    'waves/ surf'
    'ocean'
    'running water'
    'still water'
    'ice'
    'snow'
    'clouds'
    'smoke'
    'fire'
    'natural light'
    'direct sun/sunny'
    'electric/indoor lighting'
    'aged/ worn'
    'glossy'
    'matte'
    'sterile'
    'moist/ damp'
    'dry'
    'dirty'
    'rusty'
    'warm'
    'cold'
    'natural'
    'man-made'
    'open area'
    'semi-enclosed area'
    'enclosed area'
    'far-away horizon'
    'no horizon'
    'rugged scene'
    'mostly vertical components'
    'mostly horizontal components'
    'symmetrical'
    'cluttered space'
    'scary'
    'soothing'
    'stressful'


'''
class SUNDataset(Dataset):

    def __init__(self, labels_dir, images_dir, labels_mode = 'cont', mode = 'train', transform = None):
         """
        :param labels_dir: Path to the labels folder.
        :param images_dir: Path to the images folder.
        :param labels_mode = Type of labels to use
        :param mode: 'train' or 'val' or 'test'
        :param transform: Optional transform. (But make sure images are the same size if you choose to omit!)
        """
        # train/val/test
        self.mode = 'mode'
        # Labeling categories
        cat_path = os.fsencode(labels_dir + 'attributes.mat')
        self.categories = scipy.io.loadmat(cat_path)['attributes']
        if labels_mode == 'naive':
            labels_path = os.fsencode(labels_dir + 'attributeLabels_naive.mat')
        elif labels_mode == 'cont':
            labels_path = os.fsencode(labels_dir + 'attributeLabels_continuous.mat')
        elif labels_mode == 'sample':
            labels_path = os.fsencode(labels_dir + 'attributeLabels_sample.mat')
        else:
            raise NotImplementedError
        # Ground truth matrix: 14340x102
        self.attributes_matrix = scipy.io.loadmat(labels_path)['labels_cv']
        # images directory
        self.images_dir = images_dir
        image_names_path = os.fsencode(labels_dir + 'images.mat')
        image_names = scipy.io.loadmat(image_names_path)['images']
        # list of image names as string
        self.images_list = []
        for image_name in image_names:
            self.images_list.append(os.fsencode(images_dir + image_name))
        # train/val/test separation
        if self.mode == 'train':
            self.images_list = self.images_list[:11472]
            self.attributes_matrix = self.attributes_matrix[:11472,:]
        elif self.mode == 'val':
            self.images_list = self.images_list[11472:12906]
            self.attributes_matrix = self.attributes_matrix[11472:12906,:]
        elif self.mode == 'test':
            self.images_list = self.images_list[12906:]
            self.attributes_matrix = self.attributes_matrix[12906:, :]
        else:
            raise NotImplementedError
        # Transforms
        self.transform = transform
    def __len__(self):
        return len(self.images_list)
    def __getitem__(self, index):
        image = io.imread(self.images_list[index])

        if self.transform:
            image = self.transform(image)
        else:
            self.to_tensor = transforms.ToTensor()
            image = self.to_tensor(image)

        labels = np.asarray(self.attributes_matrix[index, :])
        return image, labels
