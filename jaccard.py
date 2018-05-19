import os
import scipy.io
import numpy as np
import operator

"""
Python script to compute pairwise Jaccard similarities between categories.
Note that we will exclude 'sleevelength', 'neckline' and 'category' because they have non-binary labels.
Dataset source: 
Huizhong Chen, Andrew Gallagher, and Bernd Girod, "Describing Clothing by Semantic Attributes", European Conference on 
Computer Vision (ECCV), October 2012.
Research Datasets for Image, Video, and Multimedia Systems Group at Stanford
"""


def main():
    # Labels directory
    labels_dir = './ClothingAttributeDataset/labels/'

    # List of all the labeling categories
    categories = ['black', 'blue', 'brown', 'collar', 'cyan', 'gender', 'gray', 'green',
                  'many_colors', 'necktie', 'pattern_floral', 'pattern_graphics', 'pattern_plaid',
                  'pattern_solid', 'pattern_spot', 'pattern_stripe', 'placket', 'purple', 'red', 'scarf',
                  'skin_exposure', 'white', 'yellow']

    # This will hold the ground truth labels for all images across the categories
    attributes = {}  # {'black': (1856, 1), ... }
    # np.random.seed(0)

    for cat in categories:
        dir = os.fsencode(labels_dir + cat + "_GT.mat")
        labels = scipy.io.loadmat(dir)['GT'] - 1  # (1856, 1) Change labels from (1,2) to (0,1)
        # labels[np.isnan(labels)] = np.random.randint(2)  # Note that there are nan labels in the dataset
        labels[np.isnan(labels)] = 0
        attributes[cat] = labels

    # This will hold the dictionaries that hold all jaccard similarities for a given category.
    jaccard_dict = {}  # {'black': {}, ... ,'yellow': {}} nested dictionary

    # Loop over all categories
    for cat in categories:
        # This dictionary will go into the jaccard_dict in each iteration
        jaccard = {}  # {'black': 0.41379310344827586, ... }
        # Loop over all other categories except for the current one
        for attribute in attributes:
             if attribute != cat:
                first = np.asarray(attributes[attribute], dtype=int)
                second = np.asarray(attributes[cat], dtype=int)
                p = np.count_nonzero(first & second)
                q = np.count_nonzero(first | second)
                jaccard[attribute] = round(p / q, 4)
        jaccard_dict[cat] = jaccard

    # Print out the results to console (uncomment below if you want to see all similarities)
    for cat in jaccard_dict:
        print('Category: ' + cat)
        #print(jaccard_dict[cat])
        print('Max:')
        print(max(jaccard_dict[cat].items(), key=operator.itemgetter(1)))
        print('Min:')
        print(min(jaccard_dict[cat].items(), key=operator.itemgetter(1)))
        print()

    # Let's write the jaccard similarities between gender and other categories to a .txt file
    f = open('jaccard.txt', 'w')
    for cat in categories:
        if cat != 'gender':
            f.write(cat + ': ' + '{}'.format(jaccard_dict['gender'][cat]) + '\n')
    f.close()


if __name__ == "__main__":
    main()


"""
When nan is set to 0:

Category: black
Max:
('pattern_solid', 0.3083)
Min:
('many_colors', 0.0)

Category: blue
Max:
('placket', 0.0872)
Min:
('many_colors', 0.0)

Category: brown
Max:
('collar', 0.1213)
Min:
('many_colors', 0.0)

Category: collar
Max:
('placket', 0.6511)
Min:
('pattern_graphics', 0.006)

Category: cyan
Max:
('pattern_stripe', 0.0698)
Min:
('many_colors', 0.0)

Category: gender
Max:
('pattern_solid', 0.309)
Min:
('necktie', 0.0032)

Category: gray
Max:
('placket', 0.21)
Min:
('many_colors', 0.0)

Category: green
Max:
('pattern_graphics', 0.0489)
Min:
('many_colors', 0.0)

Category: many_colors
Max:
('pattern_floral', 0.2089)
Min:
('black', 0.0)

Category: necktie
Max:
('collar', 0.2289)
Min:
('pattern_floral', 0.0)

Category: pattern_floral
Max:
('many_colors', 0.2089)
Min:
('necktie', 0.0)

Category: pattern_graphics
Max:
('many_colors', 0.1219)
Min:
('necktie', 0.0)

Category: pattern_plaid
Max:
('red', 0.125)
Min:
('pattern_floral', 0.0)

Category: pattern_solid
Max:
('placket', 0.5782)
Min:
('pattern_spot', 0.0017)

Category: pattern_spot
Max:
('white', 0.1206)
Min:
('pattern_floral', 0.0)

Category: pattern_stripe
Max:
('white', 0.1264)
Min:
('pattern_floral', 0.0)

Category: placket
Max:
('collar', 0.6511)
Min:
('pattern_graphics', 0.0055)

Category: purple
Max:
('pattern_stripe', 0.0534)
Min:
('green', 0.0)

Category: red
Max:
('pattern_plaid', 0.125)
Min:
('many_colors', 0.0)

Category: scarf
Max:
('placket', 0.1647)
Min:
('pattern_floral', 0.0)

Category: skin_exposure
Max:
('gender', 0.1745)
Min:
('necktie', 0.0)

Category: white
Max:
('gender', 0.238)
Min:
('many_colors', 0.0)

Category: yellow
Max:
('gender', 0.0437)
Min:
('cyan', 0.0)

"""

