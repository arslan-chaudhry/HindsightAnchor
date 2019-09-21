# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Define utility functions for manipulating datasets
"""
import os
import numpy as np
import sys
from copy import deepcopy
from PIL import Image


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import tarfile
import zipfile
import random


#########################################
## MNIST Utils ##########################
#########################################
def reformat_mnist(datasets):
    """
    Routine to Reformat the mnist dataset into a 3d tensor
    """
    image_size = 28 # Height of MNIST dataset
    num_channels = 1 # Gray scale
    for i in range(len(datasets)):
        sets = ["train", "validation", "test"]
        for set_name in sets:
            datasets[i]['%s'%set_name]['images'] = datasets[i]['%s'%set_name]['images'].reshape\
            ((-1, image_size, image_size, num_channels)).astype(np.float32)

    return datasets

def construct_permute_mnist(num_tasks):
    """
    Construct a dataset of permutted mnist images

    Args:
        num_tasks   Number of tasks
    Returns
        dataset     A permutted mnist dataset
    """
    # Download and store mnist dataset
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    datasets = []

    for i in range(num_tasks):
        perm_inds = range(mnist.train.images.shape[1])
        np.random.shuffle(perm_inds)
        copied_mnist = deepcopy(mnist)
        sets = ["train", "validation", "test"]
        for set_name in sets:
            this_set = getattr(copied_mnist, set_name) # shallow copy
            this_set._images = np.transpose(np.array([this_set.images[:,c] for c in perm_inds]))
            if set_name == "train":
                train = { 
                    'images':this_set._images,
                    'labels':this_set.labels,
                }
            elif set_name == "validation":
                validation = {
                    'images':this_set._images,
                    'labels':this_set.labels,
                }
            elif set_name == "test":
                test = {
                    'images':this_set._images,
                    'labels':this_set.labels,
                }
        dataset = {
            'train': train,
            'validation': validation,
            'test': test,
        }

        datasets.append(dataset)

    return datasets

def construct_split_mnist(task_labels):
    """
    Construct a split mnist dataset

    Args:
        task_labels     List of split labels

    Returns:
        dataset         A list of split datasets

    """
    # Download and store mnist dataset
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    datasets = []

    sets = ["train", "validation", "test"]

    for task in task_labels:

        for set_name in sets:
            this_set = getattr(mnist, set_name)

            global_class_indices = np.column_stack(np.nonzero(this_set.labels))
            count = 0

            for cls in task:
                if count == 0:
                    class_indices = np.squeeze(global_class_indices[global_class_indices[:,1] ==\
                                                                    cls][:,np.array([True, False])])
                else:
                    class_indices = np.append(class_indices, np.squeeze(global_class_indices[global_class_indices[:,1] ==\
                                                                                             cls][:,np.array([True, False])]))
                count += 1

            class_indices = np.sort(class_indices, axis=None)

            if set_name == "train":
                train = {
                    'images':deepcopy(mnist.train.images[class_indices, :]),
                    'labels':deepcopy(mnist.train.labels[class_indices, :]),
                }
            elif set_name == "validation":
                validation = {
                    'images':deepcopy(mnist.validation.images[class_indices, :]),
                    'labels':deepcopy(mnist.validation.labels[class_indices, :]),
                }
            elif set_name == "test":
                test = {
                    'images':deepcopy(mnist.test.images[class_indices, :]),
                    'labels':deepcopy(mnist.test.labels[class_indices, :]),
                }

        mnist2 = {
            'train': train,
            'validation': validation,
            'test': test,
        }

        datasets.append(mnist2)

    return datasets
