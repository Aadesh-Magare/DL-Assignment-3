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

def write_preds(preds, filename):
    with open(filename, 'w') as f:
        for p in preds:
            f.write(f'{p}\n')

if __name__ == "__main__":
    repo_path = os.path.dirname(os.path.abspath(__file__))

    train, dev, test = dataset.load_data()
    xt, yt = dataset.prepare_dataset(test)

    model = logistic_regression.load_model()
    preds = model.predict(xt)
    print('Test Acc LR model:', np.mean(preds == yt))
    write_preds(preds, os.path.join(repo_path, 'tfidf.txt'))
    print('Results saved to file', 'tfidf.txt')

    preds = neural_model.evaluate()
    write_preds(preds, os.path.join(repo_path, 'deep_model.txt'))
    print('Results saved to file', '"deep_model.txt')

