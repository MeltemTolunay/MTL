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
        p_dict[att] = (1.0 * np.count_nonzero(attributes[att])) / len(attributes[att])

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
                mi = p11 * np.log(p11 / (p_dict[cat] * p_dict[att])) + \
                     p10 * np.log(p10 / (p_dict[cat] * (1.0 - p_dict[att]))) +  \
                     p01 * np.log(p01 / (p_dict[att] * (1.0 - p_dict[cat]))) +  \
                     p00 * np.log(p00 / ((1.0 - p_dict[cat]) * (1.0 - p_dict[att])))
                mi_cat[att] = mi
        mi_dict[cat] = mi_cat

    for cat in mi_dict:
        print('Category: ' + cat)
        #print(mi_dict[cat])
        print('Max:')
        print(max(mi_dict[cat].items(), key=operator.itemgetter(1)))
        print('Min:')
        print(min(mi_dict[cat].items(), key=operator.itemgetter(1)))
        print()


if __name__ == "__main__":
    main()

    
'''
/Users/Meltem/anaconda3/envs/pytorch04/bin/python /Users/Meltem/PycharmProjects/MTL/mutual_info.py
/Users/Meltem/PycharmProjects/MTL/mutual_info.py:51: RuntimeWarning: divide by zero encountered in log
  p01 * np.log(p01 / (p_dict[att] * (1.0 - p_dict[cat]))) +  \
/Users/Meltem/PycharmProjects/MTL/mutual_info.py:51: RuntimeWarning: invalid value encountered in double_scalars
  p01 * np.log(p01 / (p_dict[att] * (1.0 - p_dict[cat]))) +  \
Category: black
Max:
('gray', 0.04439467144377464)
Min:
('necktie', 0.00019510222982007438)

Category: blue
Max:
('yellow', 0.07575960125132028)
Min:
('pattern_graphics', 4.5429400647043145e-05)

Category: brown
Max:
('yellow', 0.07385197000497788)
Min:
('pattern_floral', 5.97806040919524e-07)

Category: collar
Max:
('placket', 0.17121431396598197)
Min:
('yellow', 3.108309867822141e-06)

Category: cyan
Max:
('yellow', 0.09130471182458644)
Min:
('pattern_solid', 8.242643414739992e-06)

Category: gender
Max:
('collar', 0.09904029751261106)
Min:
('scarf', 0.0002129328540408917)

Category: gray
Max:
('black', 0.04439467144377464)
Min:
('white', 0.00041236323992113397)

Category: green
Max:
('yellow', 0.10585728470294549)
Min:
('pattern_stripe', 1.8638199247730206e-06)

Category: many_colors
Max:
('black', nan)
Min:
('black', nan)

Category: necktie
Max:
('gender', 0.0849145766342859)
Min:
('white', 2.1502034568505676e-05)

Category: pattern_floral
Max:
('many_colors', 0.022568754035426068)
Min:
('brown', 5.97806040919524e-07)

Category: pattern_graphics
Max:
('pattern_solid', 0.05294413124330523)
Min:
('cyan', 2.7726316233177718e-05)

Category: pattern_plaid
Max:
('pattern_solid', 0.02843550876715454)
Min:
('green', 3.162798153062772e-05)

Category: pattern_solid
Max:
('placket', 0.10837863231165343)
Min:
('cyan', 8.242643414739992e-06)

Category: pattern_spot
Max:
('pattern_solid', 0.04444884872641244)
Min:
('many_colors', 9.489074568902922e-05)

Category: pattern_stripe
Max:
('pattern_solid', 0.03735404199380941)
Min:
('green', 1.8638199247730206e-06)

Category: placket
Max:
('collar', 0.17121431396598197)
Min:
('red', 7.291739390247767e-08)

Category: purple
Max:
('gray', 0.004126473935266193)
Min:
('pattern_floral', 6.176226927232664e-06)

Category: red
Max:
('pattern_plaid', 0.009708787424077407)
Min:
('placket', 7.291739390247767e-08)

Category: scarf
Max:
('pattern_solid', 0.025655902289111212)
Min:
('yellow', 1.1531506854336343e-05)

Category: skin_exposure
Max:
('placket', 0.055811105401129116)
Min:
('green', 0.0003218475779107749)

Category: white
Max:
('cyan', 0.03702113395008931)
Min:
('collar', 8.04928714135322e-06)

Category: yellow
Max:
('green', 0.10585728470294548)
Min:
('placket', 3.484349055418002e-07)


Process finished with exit code 0

'''
