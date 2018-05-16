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

    # Numerical stability for log
    eps = 1e-9

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
                mi = p11 * np.log((p11 + eps) / (p_dict[cat] * p_dict[att])) + \
                     p10 * np.log((p10 + eps) / (p_dict[cat] * (1.0 - p_dict[att]))) +  \
                     p01 * np.log((p01 + eps) / (p_dict[att] * (1.0 - p_dict[cat]))) +  \
                     p00 * np.log((p00 + eps) / ((1.0 - p_dict[cat]) * (1.0 - p_dict[att])))
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
Category: black
Max:
('many_colors', 0.07744568982299985)
Min:
('necktie', 0.00019510622981998207)

Category: blue
Max:
('yellow', 0.07575960525132021)
Min:
('pattern_graphics', 4.54334006470032e-05)

Category: brown
Max:
('yellow', 0.07385197400497778)
Min:
('pattern_floral', 6.018060408338537e-07)

Category: collar
Max:
('placket', 0.17121431796598197)
Min:
('yellow', 3.1123098678894246e-06)

Category: cyan
Max:
('yellow', 0.09130471582458624)
Min:
('pattern_solid', 8.246643414705143e-06)

Category: gender
Max:
('collar', 0.09904030151261113)
Min:
('scarf', 0.00021293685404100236)

Category: gray
Max:
('black', 0.04439467544377466)
Min:
('white', 0.00041236723992110237)

Category: green
Max:
('yellow', 0.10585728870294545)
Min:
('pattern_stripe', 1.8678199247792635e-06)

Category: many_colors
Max:
('black', 0.07744568982299985)
Min:
('pattern_spot', 9.489474568891935e-05)

Category: necktie
Max:
('gender', 0.08491458063428586)
Min:
('white', 2.1506034568424858e-05)

Category: pattern_floral
Max:
('many_colors', 0.022568758035426066)
Min:
('brown', 6.018060408338537e-07)

Category: pattern_graphics
Max:
('pattern_solid', 0.05294413524330518)
Min:
('cyan', 2.7730316232964843e-05)

Category: pattern_plaid
Max:
('pattern_solid', 0.02843551276715444)
Min:
('green', 3.163198153040075e-05)

Category: pattern_solid
Max:
('placket', 0.10837863631165345)
Min:
('cyan', 8.246643414705143e-06)

Category: pattern_spot
Max:
('pattern_solid', 0.04444885272641234)
Min:
('many_colors', 9.489474568891935e-05)

Category: pattern_stripe
Max:
('pattern_solid', 0.0373540459938094)
Min:
('green', 1.8678199247792635e-06)

Category: placket
Max:
('collar', 0.17121431796598197)
Min:
('red', 7.691739395730642e-08)

Category: purple
Max:
('many_colors', 0.007897738648570912)
Min:
('pattern_floral', 6.180226926937851e-06)

Category: red
Max:
('pattern_plaid', 0.009708791424077408)
Min:
('placket', 7.691739395730642e-08)

Category: scarf
Max:
('pattern_solid', 0.02565590628911113)
Min:
('yellow', 1.1535506854161415e-05)

Category: skin_exposure
Max:
('placket', 0.05581110940112916)
Min:
('green', 0.00032185157791077886)

Category: white
Max:
('cyan', 0.037021137950089315)
Min:
('collar', 8.053287141397193e-06)

Category: yellow
Max:
('green', 0.10585728870294545)
Min:
('placket', 3.52434905520789e-07)

'''