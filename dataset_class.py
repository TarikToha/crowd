import csv

import numpy as np
import jenkspy as jp


def goodness_of_variance_fit(array, classes):
    # get the break points
    classes = jp.jenks_breaks(array, classes)

    # do the actual classification
    classified = np.array([classify(i, classes) for i in array])

    # max value of zones
    maxz = max(classified)

    # nested list of zone indices
    zone_indices = [[idx for idx, val in enumerate(classified) if zone + 1 == val] for zone in range(maxz)]

    # sum of squared deviations from array mean
    sdam = np.sum((array - array.mean()) ** 2)

    # sorted polygon stats
    array_sort = [np.array([array[index] for index in zone]) for zone in zone_indices]

    # sum of squared deviations of class means
    sdcm = sum([np.sum((classified - classified.mean()) ** 2) for classified in array_sort])

    # goodness of variance fit
    gvf = (sdam - sdcm) / sdam

    return gvf


def classify(value, breaks):
    for i in range(1, len(breaks)):
        if value < breaks[i]:
            return i
    return len(breaks) - 1


def count_class(array):
    gvf = 0.0
    nclasses = 1
    array = np.array(array)
    while gvf < 0.99:
        nclasses += 1
        print(nclasses)
        gvf = goodness_of_variance_fit(array, nclasses)
        print(gvf)

    return nclasses


def assign(range_array, label):
    t = len(range_array) - 1
    for r in range(t):
        if range_array[r] <= label < range_array[r + 1]:
            return r

    return t


def save_classes(classes):
    v = len(row)
    row[0].append('class_num')
    row[0].append('class_alpha')

    for value in range(1, v):
        cls = assign(classes, int(row[value][1]))
        row[value].append(cls)

        cls = chr(ord('A') + cls)
        row[value].append(cls)

    with open('combined_labels.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerows(row)

    # print(row)


row = []
value_array = []

with open('benchmark/nwpu_v1_train_512.csv') as file:
    data = csv.reader(file)
    header = next(data, None)
    row.append(header)

    for line in data:
        row.append(line)
        count = int(line[3])
        if count > 0:
            value_array.append(count)

print(len(row))
nclasses = count_class(value_array)
print(nclasses)
classes = jp.jenks_breaks(value_array, nclasses)

classes = np.array(classes)

np.set_printoptions(suppress=True)
np.save('classes.npy', classes)
print(classes)

# [   1.   12.   29.   50.   76.  108.  149.  198.  260.  341.  448.  579.
  # 764.  991. 1265. 1591.]

classes = np.load('classes.npy', allow_pickle=True)
np.set_printoptions(suppress=True)
# print(classes)

classes = np.insert(classes, 0, 0)
classes[-1] = 1600
with open('classes.txt', 'w') as out:
    out.write(str(classes))

# save_classes(classes)
print(classes)
