#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import os
import math
import numpy as np
import pandas as pd
import re
import sys

from scipy.stats import spearmanr
#import skimage

import time
import torch
import torch.hub
import torch.nn
import torch.utils
import torch.utils.data

import pickle
import random
import matplotlib

class Dataset(torch.utils.data.Dataset):
    __initialized = False
    def __init__(self, indexes, data_dict):
        """
        Args:
            indexes: shared indexes across different datasets
        """
        self.indexes = indexes
        self.indexes_dict = dict()
        for i, _ in enumerate(indexes):
            self.indexes_dict[_] = i
        
        self.keys = data_dict.keys()
        self.data_dict = data_dict
        self.__initialized = True

    def __len__(self):
        """Denotes the number of samples"""
        return len(self.indexes)
    
    def __getitem__(self, index):
        """Generate one batch of data.
        
        Returns:
            idx: indexes of samples (long)
        """
        # Generate indexes of the batch
        data_index = self.indexes[index]
        
        # Generate torch.long indexes of the batch samples
        idx = self.indexes_dict[data_index]

        # Generate data
        data = self.__data_generation(data_index)

        return data, idx
    
    def __data_generation(self, indexes):
        """Generates data containing batch_size samples.
        
        Returns:
            data: data.g in [b, n_microbes]; data.m in [b, n_metabolites]
        """
        
        data = dict()
        for key in self.keys:
            data[key] = torch.tensor(self.data_dict[key].loc[indexes, :].values)
        
        return data


class Encoder(torch.nn.Module):
    def __init__(self, net_dict, n_out_features, dropout_rate):
        """
        Args:
            net_dict: e.g., {gut_16s: [10000, 100, 100]} to define the structures
                      the keys should exist in keys for DataLoader
            n_out_features: number of output features
        """
        super(Encoder, self).__init__()
        
        self.keys = net_dict.keys()
        
        for key in self.keys:
            net_fcs = []
            structure = net_dict[key]
            net_in_shape = structure[0]
            for i, net_n_hidden_nodes in enumerate(structure[1:]):
                net_fcs.append(torch.nn.Linear(net_in_shape, net_n_hidden_nodes))
                net_fcs.append(torch.nn.BatchNorm1d(net_n_hidden_nodes))
                net_fcs.append(torch.nn.ReLU6())
                net_fcs.append(torch.nn.Dropout(p=dropout_rate))
                net_in_shape = net_n_hidden_nodes
            net_fcs.append(torch.nn.Linear(net_in_shape, n_out_features))
            setattr(self, key, torch.nn.ModuleList(net_fcs))
        
    def forward(self, data):
        """
        Args:
            data: a dictionary
        """
        f = dict()
        
        for key in self.keys:
            _f = data[key]
            net_fcs = getattr(self, key)
            for net_fc in net_fcs:
                _f = net_fc(_f)
            f[key] = _f
        
        return f

# train function
from utils_eval import AverageMeter

def train_unsup(epoch, model, criterion_gm, optimizer, train_loader, n_out_features,
          gradient_clip=10, print_hist=True, print_freq=1, device='cuda:0'):
    """
    One epoch training
    """
    model.train()
    criterion_gm.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    
    for idx, (data, index) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        batch_size = data[list(data.keys())[0]].size(0)
        index = index.to(device)
        for _ in data.keys():
            data[_] = data[_].float().to(device)

        # ===================forward=====================
        f = model(data)
        # Normalization
        multi_omics_feat = []
        for key in model.keys:
            norm = f[key].pow(2).sum(1, keepdim=True).pow(0.5)
            norm_omics = f[key].div(norm)
            multi_omics_feat.append(norm_omics)
        
        features = torch.cat((multi_omics_feat), dim=1)
        loss = criterion_gm(features.view(batch_size, len(model.keys), n_out_features))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        torch.nn.utils.clip_grad_norm_(criterion_gm.parameters(), gradient_clip)
        optimizer.step()

        # ===================meters=====================
        losses.update(loss.item(), batch_size)

        # torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if print_hist:
            if (idx + 1) % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                # 'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                # 'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t' \
                    .format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))
                # print(out_image.shape)
                sys.stdout.flush()

        # ===================debug======================
        if np.isnan(losses.val):
            print(list(model.parameters()))
            for key in model.keys:
                print(f[key])

            raise Exception("Nan detected")
            break
    
    return losses.avg

class MbSimCLRModel:
    def __init__(self, covariate, indexes, multi_omics_dict, df_with_covariates,
                 standardize=True, root_folder='./', random_seed=123, device='cuda:0'):
        self.covariate = covariate
        self.keys = multi_omics_dict.keys()
        self.indexes = indexes
        self.random_seed = random_seed
        self.device = device
        self.df_with_covariates = df_with_covariates
        self.root_folder = root_folder
        
        self.multi_omics_dict = multi_omics_dict.copy()
        if standardize:
            for key in self.keys:
                from sklearn.preprocessing import StandardScaler
                standardized_array = StandardScaler().fit_transform(multi_omics_dict[key])
                standardized_df = pd.DataFrame(standardized_array, index=multi_omics_dict[key].index, 
                                        columns=multi_omics_dict[key].columns)
                self.multi_omics_dict[key] = standardized_df
        self.indexes_no_ukn = self._remove_na_indexes()
        self.labels = df_with_covariates.loc[self.indexes_no_ukn, covariate]
        
        import matplotlib
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
    
    def _remove_na_indexes(self, remove_unknown=True):
        if remove_unknown:
            ### Remove rows with missing value
            all_labels = self.df_with_covariates.loc[self.indexes, self.covariate].values
            idx_ukn = np.where(np.isin(all_labels, ['Unknown', 'unknown']))[0]
            indexes_no_ukn = np.delete(np.array(self.indexes), idx_ukn)
        else:
            indexes_no_ukn = indexes
        return indexes_no_ukn
    
    def _stratified_split(self, train_val_p=[0.7, 0.15]):
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        unique_labels = np.unique(self.labels)
        indexes_train, indexes_val, indexes_test = [], [], []
        for one_label in unique_labels:
            onecate_idx = np.where(self.labels==one_label)[0]
            onecate_indexes = np.array(self.indexes_no_ukn)[onecate_idx]
            onecate_indexes_train = np.random.choice(onecate_indexes, int(len(onecate_indexes) * train_val_p[0]), 
                                                     replace=False)
            onecate_indexes_not_train = [_ for _ in onecate_indexes if _ not in onecate_indexes_train]
            onecate_indexes_val = np.random.choice(onecate_indexes_not_train, int(len(onecate_indexes) * train_val_p[1]), 
                                                   replace=False)
            onecate_indexes_test = [_ for _ in onecate_indexes_not_train if _ not in onecate_indexes_val]
            indexes_train.extend(onecate_indexes_train)
            indexes_val.extend(onecate_indexes_val)
            indexes_test.extend(onecate_indexes_test)
        return np.random.permutation(indexes_train), \
               np.random.permutation(indexes_val), \
               np.random.permutation(indexes_test)
    
    def initialize(self, net_dict, temperature=0.5, n_out_features=10, batch_size=32,
                   dropout_rate=0.4, weight_decay=0.01, lr=0.001, momentum=0.9):
        print('\n{}'.format(self.covariate))
        self.n_out_features = n_out_features
        self.temperature = temperature
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        #torch.set_deterministic(True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
            
        self.indexes_split = self._stratified_split()
        indexes_train, indexes_val, indexes_test = self.indexes_split
        print("n train: {}\nn val: {}\nn test: {}".format(len(indexes_train), len(indexes_val), len(indexes_test)))
        
        from supervised_loss import SupConLoss

        # Generator
        train_set = Dataset(indexes_train, self.multi_omics_dict)
        val_set = Dataset(indexes_val, self.multi_omics_dict)
        
        if len(indexes_train)%batch_size==1:
            self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, 
                                                       shuffle=True, num_workers=0, drop_last=True)
        else:
            self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, 
                                                       shuffle=True, num_workers=0)
        self.val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

        # Set model
        self.model = Encoder(net_dict = net_dict, n_out_features=n_out_features, 
                        dropout_rate=dropout_rate).to(self.device)
        self.criterion_gm = SupConLoss(temperature=temperature, base_temperature=temperature,
                                      device=self.device)

        # Set optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum,  
                                    weight_decay=weight_decay)


    def train_model(self, n_epoch=1000, gradient_clip=3, print_hist=True):
        
        indexes_train, indexes_val, indexes_test = self.indexes_split
        self.n_epoch = n_epoch
        
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(self.labels)
        self.classes = le.classes_

        # =========================
        hist = dict()
        hist['loss'] = []
        hist['val_loss'] = []
        hist['val_corr'] = []
        hist['val_p'] = []
        hist['run_time']  = []
        
        start_time = time.time()

        for epoch in range(n_epoch):
            # Train
            loss = train_unsup(epoch, self.model, self.criterion_gm, self.optimizer, 
                            self.train_loader, gradient_clip = gradient_clip, n_out_features=self.n_out_features,
                            print_freq=10, device=self.device, print_hist=print_hist)
            hist['loss'].append(loss)

            # Val
            self.model.eval()
            self.criterion_gm.eval()
            feat_multi_omics = [[] for i in self.keys]
            val_losses = AverageMeter()

            with torch.no_grad():
                for idx, (data, index) in enumerate(self.val_loader):
                    batch_size = data[list(data.keys())[0]].size(0)

                    index = index.to(self.device)
                    for _ in data.keys():
                        data[_] = data[_].float().to(self.device)

                    # ===================forward=====================
                    f = self.model(data)
                    # Normalization
                    
                    val_multi_omics_feat = []
                    for i, key in enumerate(self.model.keys):
                        norm = f[key].pow(2).sum(1, keepdim=True).pow(0.5)
                        norm_omics = f[key].div(norm)
                        val_multi_omics_feat.append(norm_omics)

                        if batch_size > 1:
                            feat_multi_omics[i].extend(f[key].cpu().numpy().squeeze())
                        else:
                            feat_multi_omics[i].append(f[key].cpu().numpy().squeeze())

                    val_features = torch.cat((val_multi_omics_feat), dim=1)
                    val_loss = self.criterion_gm(val_features.view(batch_size, len(self.model.keys), 
                                                                   self.n_out_features))
                    val_losses.update(val_loss.item(), batch_size)

                hist['val_loss'].append(val_losses.avg)
                if print_hist:
                    print('\tval_loss\t{}'.format(val_losses.avg))

            # Calculate correlation, prind and append
            first_omics = np.array(feat_multi_omics[0])
            second_omics = np.array(feat_multi_omics[1])
            corr = []
            pv = []
            for i in range(self.n_out_features):
                corr_res = spearmanr(first_omics[:, i], second_omics[:, i])
                corr.append(corr_res.correlation)
                pv.append(corr_res.pvalue)
            corr = np.average(corr)
            pv = np.median(pv) ## median of p-values
            hist['val_corr'].append(corr)
            hist['val_p'].append(pv)

        end_time = time.time()

        hist['run_time'] = end_time - start_time
        self.hist = hist
        if print_hist:
            print('Training time (covariate={}, epoch={}): {}\n'.format(self.covariate, self.n_epoch, end_time-start_time))
        
    def _plot_history(self, history, plot_attr, plot_labels, ylabel, title, save_name=None, 
                      figsize=(16,8), transparency=1, save=True, title_add_tuning_paras=True):
        
        if isinstance(plot_attr, (str, tuple)):
            plot_attr = [plot_attr]
        if isinstance(plot_labels, (str, tuple)):
            plot_labels = [plot_labels]
        
        fig = plt.figure(figsize=figsize)
        
        for attr, label in zip(plot_attr, plot_labels):
            plt.plot(np.arange(len(self.hist[attr])), self.hist[attr], 
                         label=label, alpha=transparency)

        plt.xlabel('Epoches', fontsize=20)
        plt.ylabel(ylabel, fontsize=20)
        if title_add_tuning_paras:
            title_print = '{} - {}\ndropout: {};$\\quad$SGD-wd: {};$\\quad$temp: {};$\\quad$seed: {}.'.\
                format(title, self.covariate, self.dropout_rate, self.weight_decay, self.temperature, self.random_seed)
        else:
            title_print = '{} - {}'.format(title, self.covariate)
        plt.title(title_print, fontsize=20)
        plt.legend(fontsize=20)
        plt.show()
        
        if save:
            save_name_prefix = save_name if save_name is not None else plot_attr
            figure_save_folder = os.path.join(self.root_folder, 'figures/train_MB-simCLR/{}'.format(self.covariate))
            os.makedirs(figure_save_folder, exist_ok=True)

            fig.savefig(os.path.join(figure_save_folder, '{}_{}_epoch-{}_temp-{}_dropout-{}_SGD-wd-{}_seed-{}.pdf'.\
                             format(save_name_prefix, self.covariate, self.n_epoch, self.temperature, 
                                    self.dropout_rate, self.weight_decay, self.random_seed)))
    
    def plot_training(self, save=True):
        self._plot_history(history=self.hist, plot_attr=['loss', 'val_loss'], 
                      plot_labels=['Training loss', 'Validation loss'],
                      ylabel='Loss', title='Loss curve', save_name='train_val_losses', save=save)
        self._plot_history(history=self.hist, plot_attr='val_corr', 
                      plot_labels='Correlation',
                      ylabel='Correlation', title='Correlation', save_name='val_corr', save=save)
        self._plot_history(history=self.hist, plot_attr='val_p', 
                      plot_labels='P-value',
                      ylabel='pvalue', title='P-value', save_name='val_p', save=save)

        
    def save_training(self):
        import pickle
        model_save_folder = os.path.join(self.root_folder, 'models/{}'.format(self.covariate))
        os.makedirs(model_save_folder, exist_ok=True)
        
        ### save model
        torch.save({
                    'epoch': self.n_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'history': self.hist
                    }, 
            os.path.join(model_save_folder, 'MB-simCLR_{}_epoch-{}_temp-{}_dropout-{}_SGD-wd-{}_seed-{}.pth'.\
                         format(self.covariate, self.n_epoch, self.temperature, 
                                self.dropout_rate, self.weight_decay, self.random_seed)))
        
    def load_model(self, model_path, embedding_path):
        ### If you are running on a CPU-only machine, add "map_location=torch.device('cpu')".
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.n_epoch = checkpoint['epoch']
        self.hist = checkpoint['history']
        
        with open(embedding_path, 'rb') as f:
            self.embeddings_covariates_dict = pickle.load(f)
            
    def _get_features(self, indexes):
        features_dict = dict()
        for key in self.keys:
            model_one_omics = getattr(self.model, key)
            _features = []
            for one_index in indexes:
                omics_data = self.multi_omics_dict[key].loc[one_index, :].values
                self.model.eval()
                with torch.no_grad():
                    _omics_feature = torch.tensor([omics_data]).float()
                    for i, _module in enumerate(list(model_one_omics.modules())[0]): # enumerate in ModuleList
                        _omics_feature = _module(torch.tensor([_omics_feature.squeeze().cpu().numpy()]).\
                                                   float().to(self.device))
                _features.append(_omics_feature.squeeze().cpu().numpy())
            features_dict[key] = pd.DataFrame(np.array(_features), index=indexes,
                                              columns=["Feature {}".format(i) for i in range(self.n_out_features)])
        return features_dict
    
    def save_embedding(self):
        indexes_set = (self.indexes_no_ukn,) + self.indexes_split
        embeddings_covariates_dict = dict.fromkeys(['all', 'train', 'val', 'test'])
        for which_dataset, idxes in zip(embeddings_covariates_dict.keys(), indexes_set):
            embeddings_covariates_dict[which_dataset] = dict()
            features_dict = self._get_features(idxes)
            embeddings_covariates_dict[which_dataset]['embedding'] = features_dict
            embeddings_covariates_dict[which_dataset]['covariate'] = self.df_with_covariates.loc[idxes, self.covariate]
            embeddings_covariates_dict[which_dataset]['indexes'] = idxes
        
        self.embeddings_covariates_dict = embeddings_covariates_dict
        import pickle
        embedding_save_folder = os.path.join(self.root_folder, 'embeddings/{}'.format(self.covariate))
        os.makedirs(embedding_save_folder, exist_ok=True)
        
        with open(os.path.join(embedding_save_folder, 'embeddings_{}_epoch-{}_temp-{}_dropout-{}_SGD-wd-{}_seed-{}.pkl').\
              format(self.covariate, self.n_epoch, self.temperature, 
                     self.dropout_rate, self.weight_decay, self.random_seed), 'wb') as f:
            pickle.dump(embeddings_covariates_dict, f)
        
    def _split_Xy_embedding(self, which_dataset_list=['train', 'val', 'test']):
        split_Xy_embedding = dict()
        for omics_key in self.keys:
            split_Xy_embedding[omics_key] = dict()
            for which_dataset in which_dataset_list:
                _embeddings_covariates_dict = self.embeddings_covariates_dict[which_dataset]
                temp_X = _embeddings_covariates_dict['embedding'][omics_key]
                temp_y = _embeddings_covariates_dict['covariate']
                split_Xy_embedding[omics_key][which_dataset] = temp_X, temp_y
        return split_Xy_embedding
    
    @staticmethod
    def _prediction_pipeline(predict_method, predictor_random_seed=123):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        if predict_method == 'logistic':
            from sklearn.linear_model import LogisticRegression
            predict_pipeline = make_pipeline(StandardScaler(), 
                                             LogisticRegression(random_state=predictor_random_seed, max_iter=1e4, 
                                                                penalty='elasticnet', solver='saga', l1_ratio=0.5))
        elif predict_method == 'svm':
            from sklearn.svm import SVC
            predict_pipeline = make_pipeline(StandardScaler(), 
                                             SVC(gamma='auto', random_state=predictor_random_seed, probability=True))  
        elif predict_method == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            predict_pipeline = make_pipeline(StandardScaler(), 
                                             RandomForestClassifier(random_state=predictor_random_seed))
        elif predict_method == 'elasticnet':
            pass
        else:
            raise ValueError("Please choose a prediction method from: {'logistic', 'svm', 'rf'}!")
        
        return predict_pipeline
        
    def predict_embedding(self, predict_method, predictor_random_seed=123):
        if not (hasattr(self, 'embeddings_covariates_dict')):
            raise Exception("Store embeddings before making predictions!")
        predict_pipeline = self._prediction_pipeline(predict_method=predict_method,
                                                predictor_random_seed=predictor_random_seed)
        split_Xy_embedding_dict = self._split_Xy_embedding()

        result_dict = dict()
        for omics_key in self.keys:
            result_dict[omics_key] = dict()
            X_train, y_train = split_Xy_embedding_dict[omics_key]['train']
            predict_pipeline.fit(X_train, y_train)
            for vt_key in ['val', 'test']:
                X_vt, y_vt = split_Xy_embedding_dict[omics_key][vt_key]
                
                accuracy = predict_pipeline.score(X_vt, y_vt)
                pred_value = predict_pipeline.predict(X_vt)
                pred_prob = predict_pipeline.predict_proba(X_vt)
                from sklearn.metrics import confusion_matrix
                conf_mat = pd.DataFrame(confusion_matrix(y_vt, pred_value, labels=self.classes),
                                        index=['true: {}'.format(x) for x in self.classes],
                                        columns=['pred: {}'.format(x) for x in self.classes])

                result_dict[omics_key][vt_key] = accuracy, pred_value, pred_prob, conf_mat
        return result_dict
    
    @staticmethod
    def _dim_reduction_pipeline(dim_reduction_method, dim_reduction_random_seed=123):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        if dim_reduction_method == 'pca':
            from sklearn.decomposition import PCA
            dim_reduction_pipeline = make_pipeline(StandardScaler(), 
                                                   PCA(n_components=2, random_state=dim_reduction_random_seed))
            
        elif dim_reduction_method == 't-sne':
            from sklearn.manifold import TSNE
            dim_reduction_pipeline = make_pipeline(StandardScaler(), 
                                             TSNE(n_components=2, random_state=dim_reduction_random_seed))  
        elif dim_reduction_method == 'umap':
            import umap
            dim_reduction_pipeline = make_pipeline(StandardScaler(), 
                                             umap.UMAP(n_components=2, random_state=dim_reduction_random_seed))
        else:
            raise ValueError("Please choose a dimensionality reduction method from: {'pca', 't-sne', 'umap'}!")
        
        return dim_reduction_pipeline
    
    def _scatter_2d_cate(self, score, covariate_value, title, save_folder_name=None,
                         save_file_name=None, save=True, cmap_name='Set1', fontsize=20, 
                         marker_size=60, transparency=0.85, marker_list = ["o", "s", "v", "+", "x","*"]):
        import matplotlib
        levels = np.unique(covariate_value, return_inverse=True)
        #plt.style.use('ggplot')
        plt.figure(figsize=(10, 10))
        c_n = len(levels[0])
        col_list = matplotlib.cm.get_cmap(cmap_name)(range(c_n))
        handles = []
        for i in range(c_n):
            idx = np.where(levels[1]==i)[0]
            scatter = plt.scatter(score[idx,0], score[idx,1], color=col_list[levels[1][idx]], s=marker_size, 
                                  alpha=transparency, marker=marker_list[i])
            handles.append(scatter)
        plt.title(title, fontsize=fontsize)
        legend = plt.legend(handles=handles, labels=list(levels[0]), title=self.covariate, fontsize=20)
        legend.get_title().set_fontsize('20')
        plt.xticks(fontsize=fontsize, rotation=45)
        plt.xlabel('Comp 1', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.ylabel('Comp 2', fontsize=fontsize)
        
        if save:
            if (save_folder_name is None) or (save_file_name is None):
                raise ValueError("Please specify the folder name and file name for saving!")
            
            figure_save_folder = os.path.join(self.root_folder, 
                                              'figures/{}'.format(save_folder_name))
            os.makedirs(figure_save_folder, exist_ok=True)
            plt.savefig(os.path.join(figure_save_folder, '{}.pdf'.\
                             format(save_file_name)),
                        bbox_inches='tight')
        plt.show()
        
    def dim_reduction_embedding(self, dim_reduction_method, dim_reduction_random_seed=123, fontsize=20,
                                which_dataset_list=['train', 'val', 'test'], save=True):
        if not (hasattr(self, 'embeddings_covariates_dict')):
            raise Exception("Store embeddings before performing dimensionality reduction!")
        dim_reduction_pipeline = self._dim_reduction_pipeline(dim_reduction_method=dim_reduction_method,
                                                dim_reduction_random_seed=dim_reduction_random_seed)
        split_Xy_embedding_dict = self._split_Xy_embedding(which_dataset_list=which_dataset_list)

        lower_dim_rep_dict = dict()
        for omics_key in self.keys:
            lower_dim_rep_dict[omics_key] = dict()
            for which_dataset in which_dataset_list:
                _X, _y = split_Xy_embedding_dict[omics_key][which_dataset]
                omics_trans = dim_reduction_pipeline.fit_transform(_X)
                lower_dim_rep_dict[omics_key][which_dataset] = omics_trans
                
                title_name = '{} on MB-simCLR embedding\n{} | {}'.format(dim_reduction_method.upper(), 
                                                                 omics_key, which_dataset.capitalize())
                
                self._scatter_2d_cate(score=omics_trans, covariate_value=_y, fontsize=fontsize,
                             title=title_name,
                             save_folder_name='{}_embedding/MB-simCLR_{}_epoch-{}_temp-{}_dropout-{}_SGD-wd-{}_seed-{}'.\
                                  format(dim_reduction_method, self.covariate, self.n_epoch, 
                                         self.temperature, self.dropout_rate, self.weight_decay, self.random_seed),
                                  save_file_name='comp1n2_{}_{}'.format(omics_key, which_dataset),
                             save=save)
        
        dim_reduction_save_folder = os.path.join(self.root_folder, 
                                             'outputs/{}_embedding/MB-simCLR_{}_epoch-{}_temp-{}_dropout-{}_SGD-wd-{}_seed-{}'.\
                                             format(dim_reduction_method, self.covariate, self.n_epoch, 
                                             self.temperature, self.dropout_rate, self.weight_decay, self.random_seed))
        os.makedirs(dim_reduction_save_folder, exist_ok=True)
        import pickle
        with open(os.path.join(dim_reduction_save_folder, 'comp1n2_embedding_dict.pkl').\
              format(self.covariate, self.n_epoch, self.temperature, self.weight_decay, self.random_seed), 'wb') as f:
            pickle.dump(lower_dim_rep_dict, f)
            
            