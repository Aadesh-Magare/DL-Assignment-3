import json
import os, requests
import numpy as np
import sys
from torchtext import data
from torchtext import datasets

def load_data():
    if not os.path.exists('./data/snli_1.0'):
        print('Downloading')
        r = requests.get('https://nlp.stanford.edu/projects/snli/snli_1.0.zip', allow_redirects=True)
        open('./data/snli_1.0.zip', 'wb').write(r.content)
        os.system('unzip ./data/snli_1.0.zip')

    with open('./data/snli_1.0/snli_1.0_train.jsonl') as f:
        train = np.array(list(map(lambda x: {k:v for k, v in json.loads(x).items() if k in ['sentence1', 'sentence2', 'gold_label']}, f.readlines())))

    with open('./data/snli_1.0/snli_1.0_dev.jsonl') as f:
        dev = np.array(list(map(lambda x: {k:v for k, v in json.loads(x).items() if k in ['sentence1', 'sentence2', 'gold_label']}, f.readlines())))

    with open('./data/snli_1.0/snli_1.0_test.jsonl') as f:
        test = np.array(list(map(lambda x: {k:v for k, v in json.loads(x).items() if k in ['sentence1', 'sentence2', 'gold_label']}, f.readlines())))

    return train, dev, test

# def prepare_dataset(data):
#     x = np.array(list(map(lambda x: x['sentence1'] + x['sentence2'], data)))
#     y = np.array(list(map(lambda x: x['gold_label'], data)))
#     return x, y

def prepare_dataset(data, remove_no_labels=False):
    if remove_no_labels:
        data = list(filter(lambda x: x['gold_label'] != '-', data))
    x = np.array(list(map(lambda x: ' '.join(['s1_' + s for s in x['sentence1'].split(' ')]) + ' '.join([' s2_' + s for s in x['sentence2'].split(' ')]), data)))
    y = np.array(list(map(lambda x: x['gold_label'], data)))

    return x, y

class SNLI():
	def __init__(self, bs, device):
		self.inputs = data.Field(lower=True, batch_first = True)
		self.answers = data.Field(sequential=False, unk_token = None, is_target = True)
		self.train, self.dev, self.test = datasets.SNLI.splits(self.inputs, self.answers)
		self.inputs.build_vocab(self.train, self.dev)
		self.answers.build_vocab(self.train)
		self.train_iter, self.dev_iter, self.test_iter = data.Iterator.splits((self.train, self.dev, self.test), 
            batch_size=bs, device=device)

	def vocab_size(self):
		return len(self.inputs.vocab)

	def output_dim(self):
		return len(self.answers.vocab)

	def labels(self):
		return self.answers.vocab.stoi

