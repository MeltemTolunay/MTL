import os
import scipy.io
import numpy as np
import operator
def main():
    """
    Assumes binary attributes, but can be generalized later.
    """

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

    mi_dict = {}
    p_dict = {}
    # marginal probs for all attributes
    for att in attributes:
        p_dict[cat] = (1.0 * np.count_nonzero(attributes[att])) / len(attributes[att])
    # loop over all attribute pairs
    for cat in categories:
        mi_cat = {}
        for att in attributes:
            if att != cat:
                # need both diff and tot to get joint distribution
                diff = attributes[cat] - attributes[att]
                tot = attributes[cat] + attributes[att]
                # use diff and tot to get joint distribution
                p11 = (1.0 * np.sum(tot == 2)) / len(diff)
                p00 = (1.0 * np.sum(tot == 0)) / len(diff)
                p01 = (1.0 * np.sum(diff == -1)) / len(diff)
                p10 = (1.0 * np.sum(diff == 1)) / len(diff)
                # calculate mutual information from joint and marginal
                mi = p11 * np.log(p11 / (p_dict[cat] * p_dict[att])) + p10 * np.log(p10 / (p_dict[cat] * (1.0 - p_dict[att]))) +
                    p01 *  np.log(p01 / (p_dict[att] * (1.0 - p_dict[cat]))) + p00 * np.log(p00 / ((1.0 - p_dict[cat]) * (1.0 - p_dict[att])))
                mi_cat[att] = mi
        mi_dict[cat] = mi_cat
    for cat in mi_dict:
        print('Category: ' + cat)
        #print(jaccard_dict[cat])
        print('Max:')
        print(max(mi_dict[cat].items(), key=operator.itemgetter(1)))
        print('Min:')
        print(min(mi_dict[cat].items(), key=operator.itemgetter(1)))
        print()

if __name__ == "__main__":
    main()
