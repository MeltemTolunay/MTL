import os
import scipy.io
import numpy as np
import operator

"""
Python script to compute sparsity of the label categories.
Note that we will exclude 'sleevelength', 'neckline' and 'category' because they have non-binary labels.
Dataset source: 
Huizhong Chen, Andrew Gallagher, and Bernd Girod, "Describing Clothing by Semantic Attributes", European Conference on 
Computer Vision (ECCV), October 2012.
Research Datasets for Image, Video, and Multimedia Systems Group at Stanford
"""


def main():
    # Labels directory
    labels_dir = './ClothingAttributeDataset/labels'

    # List of all the labeling categories
    categories = ['black', 'blue', 'brown', 'collar', 'cyan', 'gender', 'gray', 'green',
                  'many_colors', 'necktie', 'pattern_floral', 'pattern_graphics', 'pattern_plaid',
                  'pattern_solid', 'pattern_spot', 'pattern_stripe', 'placket', 'purple', 'red', 'scarf',
                  'skin_exposure', 'white', 'yellow']

    # This will hold the ground truth labels for all images across the categories
    attributes = {}  # {'black': (1856, 1), ... }
    directory = os.fsencode(labels_dir)
    np.random.seed(0)
    for i, file in enumerate(os.listdir(directory)):
        dir = os.path.join(directory, file)
        labels = scipy.io.loadmat(dir)['GT'] - 1  # (1856, 1) Change labels from (1,2) to (0,1)
        labels[np.isnan(labels)] = np.random.randint(2)  # Note that there are nan labels in the dataset
        attributes[categories[i]] = labels

    # This will hold the sparsity measure for a given category.
    sparsity = {}  # {'black': {}, ... ,'yellow': {}} nested dictionary

    # Loop over all categories
    for cat in categories:
        vector = attributes[cat]
        sparsity[cat] = round((len(vector) - np.count_nonzero(vector)) / len(vector), 4)

    sorted_sparsity = sorted(sparsity.items(), key=operator.itemgetter(1), reverse=True)

    # Print out the results to console
    print(sorted_sparsity)
    print('Max:')
    print(max(sparsity.items(), key=operator.itemgetter(1)))
    print('Min:')
    print(min(sparsity.items(), key=operator.itemgetter(1)))
    print()


if __name__ == "__main__":
    main()


'''
[('purple', 0.9585), ('red', 0.9499), ('pattern_spot', 0.9456), ('pattern_plaid', 0.9434), ('pattern_graphics', 0.9407), ('pattern_stripe', 0.9246), ('yellow', 0.9036), ('skin_exposure', 0.896), ('green', 0.8949), ('cyan', 0.8912), ('pattern_floral', 0.8885), ('blue', 0.8588), ('brown', 0.8491), ('many_colors', 0.8303), ('necktie', 0.8233), ('scarf', 0.7716), ('gray', 0.7538), ('white', 0.6886), ('black', 0.6659), ('collar', 0.5178), ('gender', 0.4106), ('placket', 0.3755), ('pattern_solid', 0.2376)]
Max:
('purple', 0.9585)
Min:
('pattern_solid', 0.2376)
'''

