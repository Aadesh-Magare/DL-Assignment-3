"""
Code to use the saved models for testing
"""

import numpy as np
import pdb
import os
from tqdm import tqdm

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import pickle
import dataset
import logistic_regression
from logistic_regression import StemmedCountVectorizer
import neural_model

def write_preds(preds, filename, extra=None):
    counter = 0
    size = 10000
    with open(filename, 'w') as f:
        for i in range(size):
            if extra and i in extra:
                f.write('neutral\n') #prediction where true label is -
                counter += 1
            else:
                label = preds[i-counter]
                f.write(f'{label}\n')

if __name__ == "__main__":
    repo_path = os.path.dirname(os.path.abspath(__file__))

    train, dev, test = dataset.load_data(repo_path)
    xt, yt, extra = dataset.prepare_dataset(test, remove_no_labels=True)
    model = logistic_regression.load_model(repo_path)
    preds = model.predict(xt)
    print('Test Acc LR model:', np.mean(preds == yt))
    write_preds(preds, os.path.join(repo_path, 'tfidf.txt'), extra)
    print('Results saved to file', 'tfidf.txt')

    preds = neural_model.evaluate(repo_path)
    labels = ['entailment', 'contradiction', 'neutral']
    preds = [labels[p] for p in preds]
    # print('Test Acc DM model:', np.mean(preds == yt))
    write_preds(preds, os.path.join(repo_path, 'deep_model.txt'), extra)
    print('Results saved to file', 'deep_model.txt')
