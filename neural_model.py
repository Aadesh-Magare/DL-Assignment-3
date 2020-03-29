#%%
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
from torch.utils.data.sampler import SubsetRandomSampler
import dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import confusion_matrix
try:
    import torchtext
except ImportError:
    print('torchtext not found. attempt installing.')
    os.system('pip install torchtext')

random_seed = 777
torch.manual_seed(random_seed)
np.random.seed(random_seed)

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dp):
        super(LSTM, self).__init__()
        self.embd = nn.Embedding(vocab_size, embedding_dim)
        self.proj = nn.Linear(embedding_dim, 300)
        self.dropout = nn.Dropout(dp)
        self.lstm = nn.LSTM(300, hidden_dim, bidirectional=True)
        # self.gru = nn.GRU(300, hidden_dim, bidirectional=True)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=300, nhead=3)
        # self.lstm = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.relu = nn.ReLU()
        self.out = nn.Sequential(
            nn.Linear(1200, 1024),
            self.relu,
            self.dropout,
            # nn.Linear(1024, 1024),
            # self.relu,
            # self.dropout,
            # nn.Linear(1024, 1024),
            # self.relu,
            # self.dropout,
            nn.Linear(1024, output_dim)
        )

    def forward(self, batch):
        premise = self.embd(batch.premise)
        hypothesis = self.embd(batch.hypothesis)
        premise = self.relu(self.proj(premise))
        hypothesis = self.relu(self.proj(hypothesis))
        premise, _ = self.lstm(premise)
        hypothesis, _ = self.lstm(hypothesis)
        premise = premise.sum(dim = 1)
        hypothesis = hypothesis.sum(dim = 1)
        concat = torch.cat((premise, hypothesis), 1)
        # print('concat shape', concat.shape)
        return self.out(concat)

def train_one_epoch(model, trainloader, optimizer, device):
    """ 

    Training the model using the given dataloader for 1 epoch.
    Input: Model, Dataset, optimizer, 

    """

    model.train()
    losses = []
    for batch_idx, batch in enumerate(trainloader):
        # Zero out the gradients
        optimizer.zero_grad()

        # Forward Propagation
        preds = model(batch)
        loss = F.cross_entropy(preds, batch.label)

        # backward propagation
        loss.backward()
        losses.append(loss.item())
        # Update the model parameters
        optimizer.step()

    return np.average(losses)

def test(model, testloader):

    model.eval()

    y_gt = []
    y_pred_label = []
    losses = []

    for batch_idx, batch in enumerate(testloader):
        out = model(batch)
        y_pred = F.softmax(out, dim=1)
        y_pred_label_tmp = torch.argmax(y_pred, dim=1)
        loss = F.cross_entropy(out, batch.label)
        losses.append(loss.item())

        # Add the labels
        y_gt += list(batch.label.numpy())
        y_pred_label += list(y_pred_label_tmp.numpy())

    return np.mean(losses), y_gt, y_pred_label

def evaluate():
    device = torch.device('cpu')
    ds = dataset.SNLI(bs=bs, device=device)

    model = LSTM(ds.vocab_size(), embedding_dim, hidden_dim, ds.output_dim(), dp)
    model.load_state_dict(torch.load(os.path.join(repo_path, "./models/model_lstm.pt"), map_location=device))
    loss, gt, pred = test(model, ds.test_iter)
    print("\nAccuracy on Test Data : {}\n".format(np.mean(np.array(gt) == np.array(pred))))

    cm = confusion_matrix(gt, pred)
    # print(cm)
    labels = ['-', 'contradiction', 'entailment', 'neutral']
    import seaborn as sn
    plt.figure(figsize=(20, 10))
    sn.heatmap(cm, annot=True, cbar=True, xticklabels=labels, yticklabels=labels) # font size
    plt.savefig(os.path.join(repo_path, './img/cm_lstm.jpg'))
    # plt.show()

if __name__ == "__main__":
    repo_path = os.path.dirname(os.path.abspath(__file__))

    number_epochs = 15
    bs = 512
    valid_size = 0.15
    embedding_dim = 300
    dp = 0.20
    hidden_dim = 300

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    ds = dataset.SNLI(bs=bs, device=device)
    model = LSTM(ds.vocab_size(), embedding_dim, hidden_dim, ds.output_dim(), dp)
    optimizer = optim.Adam(model.parameters(), lr=0.001)#, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=0, verbose=True)

    track_loss = []
    track_acc = []
    print('Training LSTM Model')

    for i in tqdm(range(1, number_epochs+1)):
        model.to(device)
        loss = train_one_epoch(model, ds.train_iter, optimizer, device)
        track_loss.append(loss)
        print('Loss: ', loss)
        # if not (i % 5) :
        # model.to(torch.device('cpu'))
        loss, gt, pred = test(model, ds.dev_iter)
        acc = np.mean(np.array(gt) == np.array(pred))
        print("\nAccuracy on Validation Data : {}\n".format(acc))
        scheduler.step(acc)

    # plt.figure()
    # plt.plot(track_loss)
    # plt.title("training loss NN")
    # plt.savefig(os.path.join(repo_path, "./img/training_loss_cnn.jpg"))

    # evaluate(ds) 

def save_model(model):
    torch.save(model.state_dict(), os.path.join(repo_path, "./models/model_lstm.pt"))
    print('saved trained model')

def load_model():
    model = LSTM(ds.vocab_size(), embedding_dim, hidden_dim, ds.output_dim(), dp)
    model.load_state_dict(torch.load(os.path.join(repo_path, "./models/model_lstm.pt")))
    return model

