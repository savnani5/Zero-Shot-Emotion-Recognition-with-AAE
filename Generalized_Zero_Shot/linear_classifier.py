import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import os
import numpy as np
from PIL import Image
import scipy.io
from tqdm.auto import tqdm
from time import sleep
from sklearn.preprocessing import MinMaxScaler
import argparse
import math

from dataset import dataloader, classifier_dataloader
from model import encoder_cada, decoder_cada, Classifier
import argparse

parser = argparse.ArgumentParser(description='Feature extraction for vae-gzsl')
parser.add_argument('--dataset', type=str, default='EBMDB',
                    help='Name of the dataset')
parser.add_argument('--batch_size', type=int, default=5,
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
args = parser.parse_args()
os.makedirs(args.model_path, exist_ok=True)


class gzsl_vae():
    """
    docstring for gzsl_vae
    """

    def __init__(self, args):
        self.device = torch.device(args.device)

        # LOAD DATA #############################
        self.scalar = MinMaxScaler()
        self.train_val_set = dataloader(transform=self.scalar, root=args.dataset_path,
                                         split='train_val', device=self.device)
        # train_set = dataloader(root=args.dataset_path,split='train', device=self.device)
        self.test_set_unseen = dataloader(transform=self.scalar, root=args.dataset_path, split='test_unseen',
                                           device=self.device)
        self.test_set_seen = dataloader(transform=self.scalar, root=args.dataset_path, split='test_seen',
                                         device=self.device)
        # val_set = dataloader(root=args.dataset_path,split='val', device=self.device)

        self.train_loader = data.DataLoader(self.train_val_set, batch_size=args.batch_size, shuffle=True)
        # self.testloader_unseen = data.DataLoader(self.test_set_unseen, batch_size=args.batch_size, shuffle=False) #for val
        # self.testloader_seen = data.DataLoader(self.test_set_seen, batch_size=args.batch_size, shuffle=False) #for val

        self.input_dim = self.train_val_set.__get_len__()
        self.atts_dim = self.train_val_set.__get_att_len__()
        self.num_classes = self.train_val_set.__totalClasses__()

        print(20 * '-')
        print('Input_dimension=%d' % self.input_dim)
        print('Attribute_dimension=%d' % self.atts_dim)
        print('z=%d' % args.latent_size)
        print('num_classes=%d' % self.num_classes)
        print(20 * '-')

        # INITIALIZE THE MODEL AND OPTIMIZER #####################
        self.model_encoder = encoder_cada(input_dim=self.input_dim, atts_dim=self.atts_dim, z=args.latent_size).to(
            self.device)
        self.model_decoder = decoder_cada(input_dim=self.input_dim, atts_dim=self.atts_dim, z=args.latent_size).to(
            self.device)

        learnable_params = list(self.model_encoder.parameters()) + list(self.model_decoder.parameters())
        self.optimizer = optim.Adam(learnable_params, lr=1.5e-3, betas=(0.9, 0.999), eps=1e-08,
                                    weight_decay=1e-4, amsgrad=True)

        self.classifier = Classifier(input_dim=args.latent_size, num_class=self.num_classes)
        self.cls_optimizer = optim.Adam(self.classifier.parameters(), lr=1.5e-2, betas=(0.9, 0.999),
                                        weight_decay=5e-4)

        self.best_loss_epoch_file_name = os.path.join(args.model_path, 'best_loss_epoch.txt')
        self.best_epoch = -1

        print(self.model_encoder)
        print(self.model_decoder)
        print(self.classifier)

        # LOAD PRETRAINED MODEL ########################
        self.model_name = os.path.join(args.model_path, 'checkpoint_cada_' + args.dataset)
        if args.pretrained:
            if args.model_path == '':
                print('Please provide the path of the pretrained model.')
            else:
                self.best_epoch = int(np.loadtxt(self.best_loss_epoch_file_name))
                checkpoint = torch.load(self.model_name + '_' + str(self.best_epoch) + '.pth')
                self.model_encoder.load_state_dict(checkpoint['model_encoder_state_dict'])
                self.model_decoder.load_state_dict(checkpoint['model_decoder_state_dict'])
                print('>> Pretrained model loaded!')

        # LOSS ############
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.loss_function_classifier = nn.NLLLoss()
        self.best_loss = np.inf
        self.loss_log_file = os.path.join(args.model_path, 'loss_log.txt')

        # Hyper-params #######
        self.gamma = torch.zeros(1, device=self.device).float()
        self.beta = torch.zeros(1, device=self.device).float()
        self.delta = torch.zeros(1, device=self.device).float()

    def train(self, _epoch):
        """
        This function trains the cada_vae model
        """

        batch_idx = 0
        if _epoch == 1:
            try:
                os.remove(self.loss_log_file)
            except FileNotFoundError:
                pass

        # Does not seem to work for the dataset EBMDB, hence ignoring
        if args.dataset != 'EBMDB':
            if 5 < _epoch < 21:
                self.delta += 0.54
            if 20 < _epoch < 76:
                self.gamma += 0.044
            if _epoch < 93:
                self.beta += 0.0026

        train_bar = tqdm(self.train_loader)
        self.model_encoder.train()
        self.model_decoder.train()
        train_loss = 0

        for batch_idx, (x, y, sig) in enumerate(train_bar):
            x.requires_grad = False
            sig.requires_grad = False

            z_img, z_sig, mu_x, log_var_x, mu_sig, log_var_sig = self.model_encoder(x, sig)
            recon_x, recon_sig, sig_decoder_x, x_decoder_sig = self.model_decoder(z_img, z_sig)
            # loss
            vae_reconstruction_loss = self.l1_loss(recon_x, x) + self.l1_loss(recon_sig, sig)
            cross_reconstruction_loss = self.l1_loss(x_decoder_sig, x) + self.l1_loss(sig_decoder_x, sig)
            KLD_loss = (0.5 * torch.sum(1 + log_var_x - mu_x.pow(2) - log_var_x.exp())) + (
                    0.5 * torch.sum(1 + log_var_sig - mu_sig.pow(2) - log_var_sig.exp()))
            distributed_loss = torch.sqrt(torch.sum((mu_x - mu_sig) ** 2, dim=1) + torch.sum(
                (torch.sqrt(log_var_x.exp()) - torch.sqrt(log_var_sig.exp())) ** 2, dim=1))
            distributed_loss = distributed_loss.sum()

            self.optimizer.zero_grad()

            loss = vae_reconstruction_loss - self.beta * KLD_loss
            if self.delta > 0:
                loss += distributed_loss * self.delta
            if self.gamma > 0:
                loss += cross_reconstruction_loss * self.gamma

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            train_bar.set_description('l:%.3f' % (train_loss / (batch_idx + 1)))
        batch_loss = train_loss / (batch_idx + 1)
        with open(self.loss_log_file, 'a') as lf:
            lf.writelines(str(batch_loss) + '\n')
        if batch_loss < self.best_loss:
            self.best_loss = batch_loss
            self.best_epoch = _epoch
            np.savetxt(self.best_loss_epoch_file_name, [_epoch])
            torch.save({
                'epoch': _epoch,
                'model_encoder_state_dict': self.model_encoder.state_dict(),
                'model_decoder_state_dict': self.model_decoder.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': batch_loss,
            }, self.model_name + '_' + str(self.best_epoch) + '.pth')

        # print('vae %f, da %f, ca %f'%(vae,da,ca))
        print('loss: {:.3f}\t(best: {:.3f} at epoch {:d})'.format(batch_loss, self.best_loss, self.best_epoch))

    # FEATURE EXTRACTION #######################
    def extract_features(self, _params):
        print(20 * '-')
        print('Preparing dataset for the classifier..')

        self.model_encoder.eval()
        self.model_decoder.eval()

        img_seen_feats = _params['img_seen']
        img_unseen_feats = _params['img_unseen']
        att_seen_feats = _params['att_seen']
        att_unseen_feats = _params['att_unseen']

        seen_classes = self.train_val_set.__NumClasses__()
        unseen_classes = self.test_set_unseen.__NumClasses__()

        # atts for unseen classes
        attribute_vector_unseen, labels_unseen = self.test_set_unseen.__attributeVector__()

        # for train_val features:
        features_seen = [] 
        labels_seen = []
        k = 0
        for n in seen_classes:
            per_class_features = self.train_val_set.__get_perclass_feats__(n)
            k += per_class_features.shape[0]
            repeat_factor = math.ceil(img_seen_feats / per_class_features.shape[0])
            per_class_x = np.repeat(per_class_features, repeat_factor, axis=0)
            per_class_labels = torch.from_numpy(np.repeat(n, img_seen_feats, axis=0)).long()
            seen_feats = per_class_x[:img_seen_feats].float()
            # if seen_feats.shape[0] < 200:
            # 	print(n,'-------', seen_feats.shape)
            features_seen.append(seen_feats)
            labels_seen.append(per_class_labels)
        print('Number of seen features:', k)

        tensor_seen_features = torch.cat(features_seen)
        tensor_seen_feats_labels = torch.cat(labels_seen)
        tensor_unseen_attributes = torch.from_numpy(
            np.repeat(attribute_vector_unseen, att_unseen_feats, axis=0)).float()
        tensor_unseen_labels = torch.from_numpy(np.repeat(labels_unseen, att_unseen_feats, axis=0)).long()

        test_unseen_x, test_unseen_y = self.test_set_unseen.__Test_Features_Labels__()
        test_seen_x, test_seen_y = self.test_set_seen.__Test_Features_Labels__()

        with torch.no_grad():
            z_img, z_att, mu_x, log_var_x, mu_att, log_var_att = self.model_encoder(tensor_seen_features,
                                                                                    tensor_unseen_attributes)
            z_unseen_test_img, z_unseen_test_att, mu_x_unseen, log_var_x, mu_att, log_var_att = self.model_encoder(
                test_unseen_x, tensor_unseen_attributes)
            z_seen_test_img, z_unseen_test_att, mu_x_seen, log_var_x, mu_att, log_var_att = self.model_encoder(
                test_seen_x, tensor_unseen_attributes)

            train_features = torch.cat((z_att, z_img))
            train_labels = torch.cat((tensor_unseen_labels, tensor_seen_feats_labels))

        test_unseen_y = torch.squeeze(test_unseen_y)
        test_seen_y = torch.squeeze(test_seen_y)

        print('>> Extraction of train_val, test seen, and test unseen features are complete!')
        print(train_features.shape, train_labels.shape)
        # return train_features, train_labels, z_unseen_test_img, test_unseen_Y, z_seen_test_img, test_seen_Y

        return train_features, train_labels, z_unseen_test_img, test_unseen_y, mu_x_seen, test_seen_y

    # TRAINING THE CLASSIFIER #######################
    def train_classifier(self, _epochs):
        train_features, train_labels, test_unseen_features,\
            test_unseen_labels, test_seen_features, test_seen_labels = self.extract_features(params)
        np.save('./test_novel_Y.npy', test_unseen_labels)

        self.cls_train_data = classifier_dataloader(features_img=train_features, labels=train_labels,
                                                     device=self.device)
        self.cls_train_loader = data.DataLoader(self.cls_train_data, batch_size=6, shuffle=True)

        self.cls_test_unseen = classifier_dataloader(features_img=test_unseen_features, labels=test_unseen_labels,
                                                      device=self.device)
        self.cls_test_unseen_loader = data.DataLoader(self.cls_test_unseen, batch_size=6, shuffle=False)
        self.test_unseen_target_classes = self.cls_test_unseen.__targetClasses__()

        self.cls_test_seen = classifier_dataloader(features_img=test_seen_features, labels=test_seen_labels,
                                                    device=self.device)
        self.cls_test_seen_loader = data.DataLoader(self.cls_test_seen, batch_size=6, shuffle=False)
        self.test_seen_target_classes = self.cls_test_seen.__targetClasses__()

        best_H = -1
        best_seen = 0
        best_unseen = 0

        # TRAINING ####################
        for _epoch in range(1, _epochs + 1):
            print('Training: Epoch - ', _epoch)
            self.classifier.train()
            train_bar_cls = tqdm(self.cls_train_loader)
            train_loss = 0
            for batch_idx, (x, y) in enumerate(train_bar_cls):
                output = self.classifier(x)
                loss = self.loss_function_classifier(output, y)
                self.cls_optimizer.zero_grad()
                loss.backward()
                self.cls_optimizer.step()
                train_loss += loss.item()
                train_bar_cls.set_description('l:%.3f' % (train_loss / (batch_idx + 1)))

            # VALIDATION ##################
            accu_unseen = 0
            accu_seen = 0

            def val_gzsl(test_bar_cls):
                with torch.no_grad():
                    self.classifier.eval()
                    print('**Validation**')
                    pred = []
                    target = []
                    for _batch_idx, (_x, _y) in enumerate(test_bar_cls):
                        _output = self.classifier(_x)
                        output_data = torch.argmax(_output.data, 1)
                        pred.append(output_data)
                        target.append(_y)
                    predictions = torch.cat(pred)
                    targets = torch.cat(target)
                    return predictions, targets

            test_bar_cls_unseen = tqdm(self.cls_test_unseen_loader)
            test_bar_cls_seen = tqdm(self.cls_test_seen_loader)

            pred_unseen, target_unseen = val_gzsl(test_bar_cls_unseen)
            pred_seen, target_seen = val_gzsl(test_bar_cls_seen)

            # ACCURACY METRIC ##################
            def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes):
                per_class_accuracies = torch.zeros(target_classes.shape[0]).float().to(self.device)
                predicted_label = predicted_label.to(self.device)
                for i in range(target_classes.shape[0]):
                    is_class = test_label == target_classes[i]
                    per_class_accuracies[i] = torch.div(
                        (test_label[is_class] == predicted_label[is_class]).float().sum(),
                        torch.sum(is_class).float())
                return per_class_accuracies.mean()

            ##################################

            '''
            For NLL loss the labels are mapped from 0-n, map them back to 1-n for calculating accuracies.
            No need to do this for the dataset EBMDB
            '''
            if args.dataset != 'EBMDB':
                target_unseen = target_unseen + 1
                pred_unseen = pred_unseen + 1
                target_seen = target_seen + 1
                pred_seen = pred_seen + 1
            ##################################

            accu_unseen = compute_per_class_acc_gzsl(target_unseen, pred_unseen, self.test_unseen_target_classes)
            accu_seen = compute_per_class_acc_gzsl(target_seen, pred_seen, self.test_seen_target_classes)

            if (accu_seen + accu_unseen) > 0:
                H = (2 * accu_seen * accu_unseen) / (accu_seen + accu_unseen)
            else:
                H = 0

            if H > best_H:
                best_seen = accu_seen
                best_unseen = accu_unseen
                best_H = H

            print(20 * '-')
            print('Epoch:', _epoch)
            print('unseen accu, seen accu, harmonic_mean = {:.4f}, {:.4f}, {:.4f}'.
                  format(accu_unseen, accu_seen, H))
            print('Best: unseen accu, seen accu, harmonic_mean = {:.4f}, {:.4f}, {:.4f}'.
                  format(best_seen, best_unseen, best_H))
            print(20 * '-')

        return best_seen, best_unseen, best_H


if __name__ == '__main__':
    model = gzsl_vae(args)
    if not args.pretrained:
        epochs = 100 #5000
        for epoch in range(1, epochs + 1):
            print('epoch:', epoch)
            model.train(epoch)
    
    # CLASSIFIER
    params = {'img_seen': 720,
              'img_unseen': 0,
              'att_seen': 0,
              'att_unseen': 720}
    num_epochs = 40
    s, u, h = model.train_classifier(num_epochs)

