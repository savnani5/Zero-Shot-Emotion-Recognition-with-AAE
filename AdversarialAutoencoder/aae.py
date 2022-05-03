import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import os
import matplotlib.pyplot as plt
import time
from viz import *
import numpy as np
from PIL import Image
import scipy.io
from tqdm.auto import tqdm
from time import sleep
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import scipy.spatial.distance
import argparse
import math
from gensim.models import KeyedVectors

from dataset import dataloader, classifier_dataloader
# from model import Q_net, P_net, D_net_cat, D_net_gauss, sample_categorical, report_loss,\
                                        # create_latent, get_categorical, classification_accuracy
import argparse

parser = argparse.ArgumentParser(description='Feature extraction for vae-gzsl')
parser.add_argument('--dataset', type=str, default='EBMDB',
                    help='Name of the dataset')
parser.add_argument('--batch_size', type=int, default=6,
                    help='The batch size')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs')
parser.add_argument('--latent_size', type=int, default=16,
                    help='size of the latent vector')
parser.add_argument('--dataset_path', type=str, default='../feature_data/',
                    help='Name of the dataset')
parser.add_argument('--model_path', type=str, default='../models/gzsl',
                    help='path of pretrained model')
parser.add_argument('--device', type=str, default='cpu',
                    help='cuda or cpu')
parser.add_argument('--pretrained', default=False, action='store_true', help='Load pretrained weights')
parser.add_argument('--word_vec_loc', default='../feature_data/GoogleNews-vectors-negative300.bin', action='store_true', help='Load pretrained weights')
args = parser.parse_args()
cuda = torch.cuda.is_available()
os.makedirs(args.model_path, exist_ok=True)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
n_classes = 11
z_dim = 16
X_dim = 64
y_dim = 300
train_batch_size = args.batch_size
valid_batch_size = args.batch_size
N = 100

model = KeyedVectors.load_word2vec_format(args.word_vec_loc, binary=True)
emotions = ['joy','relief','pride','shame','anger','surprise','amusement','sadness','fear','neutral','disgust']
# emotions = ['relief','pride','shame','anger','surprise','fear']
emb = []
for em in emotions:
    vector = model[em]
    emb.append(vector)
emb = np.asarray(emb)

############################################################
class Q_net(nn.Module):
    def __init__(self):
        super(Q_net, self).__init__()
        self.lin1 = nn.Linear(X_dim, N)
        self.lin2 = nn.Linear(N, N)
        # Gaussian code (z)
        self.lin3gauss = nn.Linear(N, z_dim)
        # Categorical code (y)
        self.lin3cat = nn.Linear(N, y_dim)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        # x = F.dropout(self.lin1(x), p=0.25, training=self.training)
        # x = F.relu(self.lin1(x))
        x = self.lrelu(self.lin1(x))
        # x = F.dropout(self.lin2(x), p=0.25, training=self.training)
        x = F.relu(self.lin2(x))
        xgauss = self.lin3gauss(x)
        xcat = self.lin3cat(x)#F.softmax(self.lin3cat(x))

        return xcat, xgauss


# Decoder
class P_net(nn.Module):
    def __init__(self):
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(z_dim + y_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, X_dim)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.lin1(x)
        # x = F.dropout(x, p=0.25, training=self.training)
        x = self.lrelu(x)#F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        # x = F.dropout(x, p=0.25, training=self.training)
        x = self.lin3(x)
        return F.sigmoid(x)


# Discriminator networks
class D_net_cat(nn.Module):
    def __init__(self):
        super(D_net_cat, self).__init__()
        self.lin1 = nn.Linear(y_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        # x = F.relu(x)
        x = self.lrelu(x)
        x = self.lin3(x)
        return F.sigmoid(x)


class D_net_gauss(nn.Module):
    def __init__(self):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(z_dim, 100)
        self.lin2 = nn.Linear(100, 100)
        self.lin3 = nn.Linear(N, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        # x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(self.lin1(x))
        # x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        # x = F.relu(self.lin2(x))
        x = self.lin2(x)
        x = self.lrelu(x)
        x = self.lin3(x)
        return F.sigmoid(x)

####################
# Utility functions
####################
def sample_categorical(batch_size, n_classes=10):
    '''
     Sample from a categorical distribution
     of size batch_size and # of classes n_classes
     return: torch.autograd.Variable with the sample
    '''
    cat = np.random.randint(0, n_classes, batch_size)
    cat = np.eye(n_classes)[cat].astype('float32')
    cat = torch.from_numpy(cat)
    return Variable(cat)

def getNegative(labl):
    bsize = labl.shape[0]
    negative = torch.zeros((bsize, 300,10))
    for i in range(bsize):
        vecList = []
        for j in range(len(emb)):
            if j !=labl[i]:
                vecList.append(emb[j])
        vecList = np.asarray(vecList)
        vecList = vecList.T
        vecList = torch.from_numpy(vecList)
        negative[i,:,:] = vecList
    return negative.cuda()

def report_loss(epoch, D_loss_gauss, G_loss, recon_loss):
    '''
    Print loss
    '''
    print('Epoch-{}; D_loss_gauss: {:.4f}; G_loss: {:.4f}; recon_loss: {:.4f}'.format(epoch,
                                                                                    D_loss_gauss.data,
                                                                                    G_loss.data,
                                                                                    recon_loss.data))
def getSimilarity(arr,emb):
    lab = []
    for i in range(len(arr)):
        vec = arr[i]
        acc = []
        for j in range(len(emb)):
            simil = 1 - scipy.spatial.distance.cosine(vec, emb[j])
            acc.append(simil)
        temp = max(acc)
        res = acc.index(temp)

        # res = [l for l, k in enumerate(acc) if k == temp]
        lab.append(res)
    lab = np.asarray(lab)
    # temp_lab = np.zeros(lab.shape[0])
    # for idx in range(len(temp_lab)):
    #     temp_lab[idx] = mapping_seen[lab[idx].item()]
    # temp_lab = temp_lab.astype(int)
    return lab    

def classification_accuracy(Q, data_loader):
    Q.eval()
    labels = []
    predictions = []
    test_loss = 0
    correct = 0

    for batch_idx, (X, labl, target) in enumerate(data_loader):
        # X = X * 0.3081 + 0.1307
        X.resize_(data_loader.batch_size, X_dim)
        X, target = Variable(X), Variable(target)
        if cuda:
            X, target = X.cuda(), target.cuda()
        ###################
        output = Q(X)[0]
        pred = output.data
        predNP = pred.cpu().numpy()
        lab = getSimilarity(predNP,emb)
        predictions.extend(lab.tolist())

        labl = labl.cpu().numpy()
        labl = np.reshape(labl,(len(labl),1))
        labels.extend(labl.tolist())
        
        correct += (lab==labl).sum()

    labels = list(map(str, labels))
    predictions = list(map(str, predictions))
    cm = confusion_matrix(labels, predictions)
    print(cm)
    return 100. * correct / len(data_loader.dataset)
############################################################

def load_data(args):
    print('loading data!')
    device = torch.device(args.device)

    # LOAD DATA #############################
    scalar = MinMaxScaler()
    train_val_set = dataloader(transform=scalar, root=args.dataset_path,
                                         split='train_val', device=device)
  
    test_set_unseen = dataloader(transform=scalar, root=args.dataset_path, split='test_unseen',
                                           device=device)
    test_set_seen = dataloader(transform=scalar, root=args.dataset_path, split='test_seen',
                                         device=device)
    train_loader = data.DataLoader(train_val_set, batch_size=args.batch_size, shuffle=True)
    test_set_seen = data.DataLoader(test_set_seen, batch_size=2, shuffle=True)

    return train_loader, test_set_seen, test_set_unseen,train_val_set


def train(P, Q, D_cat, D_gauss, P_decoder, Q_encoder, Q_semi_supervised, \
    Q_generator, D_cat_solver, D_gauss_solver, train_labeled_loader, train_unlabeled_loader):
    '''
    Train procedure for one epoch.
    '''
    TINY = 1e-6
    # Set the networks in train mode (apply dropout when needed)
    Q.train()
    P.train()
    D_cat.train()
    D_gauss.train()

    if train_unlabeled_loader is None:
        train_unlabeled_loader = train_labeled_loader

    # Loop through the labeled and unlabeled dataset getting one batch of samples from each
    # The batch size has to be a divisor of the size of the dataset or it will return
    # invalid samples
    for (X_l, Lbl,target_l) in train_labeled_loader:

        for X, target in [(X_l, target_l)]:
            # if target[0] == -1:
            #     labeled = False
            # else:
            #     labeled = True

            # Load batch and normalize samples to be between 0 and 1
            # X = X * 0.3081 + 0.1307
            X.resize_(train_batch_size, X_dim)
            X, target = Variable(X), Variable(target)
            if cuda:
                X, target = X.cuda(), target.cuda()

            # Init gradients
            negative = getNegative(Lbl)
            P.zero_grad()
            Q.zero_grad()
            D_cat.zero_grad()
            D_gauss.zero_grad()

            #######################
            # Reconstruction phase
            #######################
            # if not labeled:
            z_sample = torch.cat(Q(X), 1)
            X_sample = P(z_sample)

            # recon_loss = F.binary_cross_entropy(X_sample + TINY, X.resize(train_batch_size, X_dim) + TINY)
            # recon_loss = nn.MSELoss()(X_sample + TINY, X.resize(train_batch_size, X_dim) + TINY)
            # recon_loss = 1. - nn.CosineSimilarity(dim=1,eps=1e-6)(X_sample + TINY, X.resize(train_batch_size, X_dim) + TINY).mean()
            recon_loss = F.binary_cross_entropy(X_sample + TINY, X.resize(train_batch_size, X_dim) + TINY)
            # print(recon_loss)
            recon_loss = recon_loss
            recon_loss.backward()
            P_decoder.step()
            Q_encoder.step()

            P.zero_grad()
            Q.zero_grad()
            D_cat.zero_grad()
            D_gauss.zero_grad()
            recon_loss = recon_loss
            #######################
            # Regularization phase
            #######################
            # Discriminator
            Q.eval()
            z_real_cat = Variable(target)#sample_categorical(train_batch_size, n_classes=n_classes)
            z_real_gauss = Variable(torch.randn(train_batch_size, z_dim))
            if cuda:
                z_real_cat = z_real_cat.cuda()
                z_real_gauss = z_real_gauss.cuda()

            z_fake_cat, z_fake_gauss = Q(X)

            D_real_cat = D_cat(z_real_cat)
            D_real_gauss = D_gauss(z_real_gauss)
            D_fake_cat = D_cat(z_fake_cat)
            D_fake_gauss = D_gauss(z_fake_gauss)

            # D_loss_cat = -torch.mean(torch.log(D_real_cat + TINY) + torch.log(1 - D_fake_cat + TINY))
            # D_loss_cat = 
            D_loss_gauss = -torch.mean(torch.log(D_real_gauss + TINY) + torch.log(1 - D_fake_gauss + TINY))

            D_loss = D_loss_gauss# +  D_loss_cat 
            D_loss = D_loss

            D_loss.backward()
            # D_cat_solver.step()
            D_gauss_solver.step()

            P.zero_grad()
            Q.zero_grad()
            D_cat.zero_grad()
            D_gauss.zero_grad()

            # Generator
            Q.train()
            # with torch.autograd.detect_anomaly():

            z_fake_cat, z_fake_gauss = Q(X)

            D_fake_cat = D_cat(z_fake_cat)
            D_fake_gauss = D_gauss(z_fake_gauss)

            # G_loss = -torch.mean(torch.log(D_fake_cat + TINY)) - torch.mean(torch.log(D_fake_gauss + TINY))
            triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
            G_triplet_loss = 0.

            for indx in range(10):
                out = triplet_loss(z_fake_cat, target, negative[:,:,indx])
                # G_triplet_loss[indx] = out
                G_triplet_loss  += out
            # out0 = triplet_loss(z_fake_cat, target, negative[:,:,0])
            # out1 = triplet_loss(z_fake_cat, target, negative[:,:,1])
            # GT_loss = G_triplet_loss.mean()
            # print(GT_loss)

            G_loss = - torch.mean(torch.log(D_fake_gauss + TINY)) + G_triplet_loss
            G_loss = G_loss
            G_loss.backward()
            Q_generator.step()

            P.zero_grad()
            Q.zero_grad()
            D_cat.zero_grad()
            D_gauss.zero_grad()

            #######################
            # Semi-supervised phase
            #######################
            
            # pred, _ = Q(X)
            # class_loss = torch.nn.MSELoss()(pred, target)#F.cross_entropy(pred, target)
            # class_loss.backward()
            # Q_semi_supervised.step()

            # P.zero_grad()
            # Q.zero_grad()
            # D_cat.zero_grad()
            # D_gauss.zero_grad()

    return D_loss_gauss, G_loss, recon_loss#, class_loss


#     return best_seen, best_unseen, best_H
def generate_model(train_labeled_loader, train_unlabeled_loader, valid_loader):
    torch.manual_seed(10)

    if cuda:
        Q = Q_net().cuda()
        P = P_net().cuda()
        D_cat = D_net_cat().cuda()
        D_gauss = D_net_gauss().cuda()
    else:
        Q = Q_net()
        P = P_net()
        D_gauss = D_net_gauss()
        D_cat = D_net_cat()

    # Set learning rates
    gen_lr = 0.0006
    semi_lr = 0.001
    reg_lr = 0.0001

    # Set optimizators
    P_decoder = optim.Adam(P.parameters(), lr=gen_lr)
    Q_encoder = optim.Adam(Q.parameters(), lr=gen_lr)

    Q_semi_supervised = optim.Adam(Q.parameters(), lr=semi_lr)

    Q_generator = optim.Adam(Q.parameters(), lr=reg_lr, weight_decay=5e-4)
    D_gauss_solver = optim.Adam(D_gauss.parameters(), lr=reg_lr)
    D_cat_solver = optim.Adam(D_cat.parameters(), lr=reg_lr)

    start = time.time()
    for epoch in range(args.epochs):
        D_loss_gauss, G_loss, recon_loss = train(P, Q, D_cat,
                                                            D_gauss, P_decoder,
                                                            Q_encoder, Q_semi_supervised,
                                                            Q_generator,
                                                            D_cat_solver, D_gauss_solver,
                                                            train_labeled_loader,
                                                            None)
                                                            # train_unlabeled_loader)
        if epoch % 10 == 0:
            train_acc = classification_accuracy(Q, train_labeled_loader)
            val_acc = classification_accuracy(Q, train_unlabeled_loader)
            report_loss(epoch,D_loss_gauss, G_loss, recon_loss)
            # print('Classification Loss: {:.3}'.format(class_loss.data))
            print('Train accuracy: {} %'.format(train_acc))
            print('Validation accuracy: {} %'.format(val_acc))
    end = time.time()
    print('Training time: {} seconds'.format(end - start))

    return Q, P

if __name__ == '__main__':
    train_labeled_loader, train_unlabeled_loader, valid_loader, train_val_set = load_data(args)
    params = {'img_seen': 720,
              'img_unseen': 0,
              'att_seen': 0,
              'att_unseen': 720}
    Q, P = generate_model(train_labeled_loader, train_unlabeled_loader, valid_loader)
    # getFeatures(params,valid_loader,train_val_set)
