import os
import scipy.io
import numpy as np
import operator

"""
Python script to compute pairwise Jaccard similarities between categories.
Note that we will exclude 'sleevelength', 'neckline' and 'category' because they have non-binary labels.
"""


def main():
    # Labels directory
    labels_dir = './ClothingAttributeDataset/labels'

    # List of all the labeling categories
    categories = ['black', 'blue', 'brown', 'category', 'collar', 'cyan', 'gender', 'gray', 'green',
                  'many_colors', 'neckline', 'necktie', 'pattern_floral', 'pattern_graphics', 'pattern_plaid',
                  'pattern_solid', 'pattern_spot', 'pattern_stripe', 'placket', 'purple', 'red', 'scarf',
                  'skin_exposure', 'sleevelength', 'white', 'yellow']

    # This will hold the ground truth labels for all images across the categories
    attributes = {}  # {'black': (1856, 1), ... }
    directory = os.fsencode(labels_dir)
    for i, file in enumerate(os.listdir(directory)):
        dir = os.path.join(directory, file)
        if categories[i] != 'sleevelength' and categories[i] != 'neckline' and categories[i] != 'category':
            labels = scipy.io.loadmat(dir)['GT'] - 1  # (1856, 1)
            labels[np.isnan(labels)] = 0
            attributes[categories[i]] = labels

    # This will hold the dictionaries that hold all jaccard similarities for a given category.
    jaccard_dict = {}  # {'black': {}, ... ,'yellow': {}} nested dictionary

    # Loop over all categories
    for cat in categories:
        if cat != 'sleevelength' and cat != 'neckline' and cat != 'category':
            # This dictionary will go into the jaccard_dict in each iteration
            jaccard = {}  # {'black': 0.41379310344827586, ... }
            # Loop over all other categories except for the current one
            for attribute in attributes:
                 if attribute != cat:
                    diff = attributes[attribute] - attributes[cat]
                    intersection = len(diff) - np.count_nonzero(diff)
                    jaccard[attribute] = round(intersection / len(diff), 4)
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


if __name__ == "__main__":
    main()


"""
Category: black
Max:
('pattern_spot', 0.6589)
Min:
('gender', 0.4677)

Category: blue
Max:
('yellow', 0.8863)
Min:
('placket', 0.4079)

Category: brown
Max:
('yellow', 0.8788)
Min:
('gender', 0.4256)

Category: collar
Max:
('placket', 0.7662)
Min:
('gender', 0.2775)

Category: cyan
Max:
('pattern_floral', 0.9176)
Min:
('placket', 0.3885)

Category: gender
Max:
('skin_exposure', 0.5361)
Min:
('collar', 0.2775)

Category: gray
Max:
('yellow', 0.7834)
Min:
('gender', 0.4488)

Category: green
Max:
('pattern_floral', 0.9256)
Min:
('placket', 0.3815)

Category: many_colors
Max:
('pattern_floral', 0.9041)
Min:
('placket', 0.3427)

Category: necktie
Max:
('yellow', 0.8534)
Min:
('gender', 0.3346)

Category: pattern_floral
Max:
('yellow', 0.9278)
Min:
('placket', 0.3491)

Category: pattern_graphics
Max:
('yellow', 0.9111)
Min:
('placket', 0.3238)

Category: pattern_plaid
Max:
('red', 0.917)
Min:
('pattern_solid', 0.3842)

Category: pattern_solid
Max:
('placket', 0.6816)
Min:
('many_colors', 0.3475)

Category: pattern_spot
Max:
('purple', 0.9138)
Min:
('placket', 0.3513)

Category: pattern_stripe
Max:
('purple', 0.8949)
Min:
('placket', 0.3658)

Category: placket
Max:
('collar', 0.7662)
Min:
('skin_exposure', 0.3028)

Category: purple
Max:
('pattern_floral', 0.9246)
Min:
('placket', 0.3739)

Category: red
Max:
('yellow', 0.9181)
Min:
('placket', 0.3879)

Category: scarf
Max:
('yellow', 0.8508)
Min:
('gender', 0.4461)

Category: skin_exposure
Max:
('pattern_floral', 0.8825)
Min:
('placket', 0.3028)

Category: white
Max:
('pattern_spot', 0.7602)
Min:
('pattern_solid', 0.4149)

Category: yellow
Max:
('pattern_floral', 0.9278)
Min:
('placket', 0.3761)
"""
