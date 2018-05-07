import os
import scipy.io
import numpy as np
"""
Python script to compute Jaccard similarities between gender and all others 
"""

categories = ['black', 'blue', 'brown', 'category', 'collar', 'cyan', 'gender', 'gray', 'green',
              'many_colors', 'neckline', 'necktie', 'pattern_floral', 'pattern_graphics', 'pattern_plaid',
              'pattern_solid', 'pattern_spot', 'pattern_stripe', 'placket', 'purple', 'red', 'scarf',
              'skin_exposure', 'sleevelength', 'white', 'yellow']

attributes = {}
jaccard = {}

labels_dir = './ClothingAttributeDataset/labels'

directory = os.fsencode(labels_dir)
for i, file in enumerate(os.listdir(directory)):
    dir = os.path.join(directory, file)
    if categories[i] != 'sleevelength' and categories[i] != 'neckline' and categories[i] != 'category':
        attributes[categories[i]] = scipy.io.loadmat(dir)['GT'] - 1  # (1856, 1)

gender = attributes['gender']

for attribute in attributes:
    diff = attributes[attribute] - gender
    intersection = len(diff) - np.count_nonzero(diff)
    jaccard[attribute] = intersection / len(diff)

print(jaccard)
