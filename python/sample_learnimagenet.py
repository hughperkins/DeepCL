# quick example of loading data from imagenet, and feeding it to the net for training
# it can obviously be improved radically in various ways, eg by applying affine
# transformations on the incoming data, instead of just cutting a patch from topleft,
# randomized sampling of the data
# etc

# assumes that data is stored in directory data_dir
# each category is stored in a single subdirectory of data_dir
# names of each category directory is arbitrary
# each point is stored as a single jpeg file, with suffix '.jpeg', 
# case-insensitive

from __future__ import print_function
import PyDeepCL
import sys
import os
import os.path
import random

training_data = [] # tuples of (directory, filename, category)
validation_data = [] # tuples of (directory, filename, category)

local_random = random.Random()
local_random.seed(0)

def scan_subdir(category, target_dir, max_images_per_category):
    count = 0
    for filename in os.listdir(target_dir):
        filepath = target_dir + '/' + filename
        if os.path.isfile(filepath) and filename.lower().endswith('.jpeg'):
            sample = (os.path.basename(target_dir), filename, category)
            # this is not the best way to do the split, we can both think of better
            # but it's good enough for a simple prototype
            if local_random.random() < validation_fraction:
                validation_data.append(sample)
            else:
                training_data.append(sample)
            count += 1
            if count >= max_images_per_category:
                break
    print(target_dir, 'count', count)
    return count

def scan_images(data_dir, max_categories, max_images_per_category):
    num_categories_scanned = 0
    count = 0
    subdirs = os.listdir(data_dir)
    for subdir in subdirs:
        if os.path.isdir(data_dir + '/' + subdir):
            count += scan_subdir(
                num_categories_scanned + 1,
                data_dir + '/' + subdir,
                max_images_per_category=max_images_per_category)
            num_categories_scanned += 1
            if num_categories_scanned >= max_categories:
                break
    print('total count', count)
    print('training_data', training_data)
    print('validation_data', validation_data)
    return count

def run(data_dir, max_categories, max_images_per_category):
    # num_images = count_images(data_dir, max_categories, max_images_per_category)
    num_images = scan_images(data_dir, max_categories, max_images_per_category)
    cl = PyDeepCL.EasyCL()
    net = PyDeepCL.NeuralNet(cl)

data_dir = '/norep/data/imagenet/small'
max_categories = 3
max_images_per_category = 5
validation_fraction = 0.3
run(data_dir=data_dir, max_categories=max_categories,
    max_images_per_category=max_images_per_category)

