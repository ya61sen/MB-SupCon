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


def plot_loss(data, plot_attr, save_folder, fig_save_name, title, figsize=(16, 8), transparency=0.8, xlab='Epochs', ylab='Loss'):
    import os
    fig = plt.figure(figsize=figsize)
    for i in range(data.shape[1]):
        plt.plot(np.arange(len(data.T[plot_attr][i])), data.T[plot_attr][i], 
                 label=data.columns[i], alpha=transparency)
    plt.xlabel(xlab, fontsize=20)
    plt.ylabel(ylab, fontsize=20)
    plt.title(title, fontsize=20)
    plt.legend(fontsize=20)
    plt.show()
    
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    fig.savefig(os.path.join(save_folder, '{}_{}.pdf'.format(fig_save_name, plot_attr)))


def plot_training(covariate, results, model_name):
    
    fig = plt.figure(figsize=(16,8))

    plt.plot(np.arange(len(results.loc['gm_loss',:][0])), results.loc['gm_loss',:][0], 
                 label='Training loss', alpha=0.8)

    plt.plot(np.arange(len(results.loc['val_gm_loss',:][0])), results.loc['val_gm_loss',:][0], 
                 label='Validation loss', alpha=0.8)

    plt.xlabel('Epoches', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title('Loss - {}'.format(covariate), fontsize=20)
    plt.legend(fontsize=20)
    plt.show()

    save_folder = './figures/{}_{}'.format(model_name, covariate)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    fig.savefig(os.path.join(save_folder, '{}_train_val_losses.pdf'.format(model_name)))
    
    plot_loss(results, 'val_gm_corr', save_folder=save_folder, fig_save_name = model_name, 
              title='Correlation - {}'.format(covariate), ylab='correlation')
    plot_loss(results, 'val_gm_p', save_folder=save_folder, fig_save_name = model_name, 
              title='p-value - {}'.format(covariate), ylab='pvalue')
    
def plot_training_unsup(results, model_name):
    
    fig = plt.figure(figsize=(16,8))

    plt.plot(np.arange(len(results.loc['gm_loss',:][0])), results.loc['gm_loss',:][0], 
                 label='Training loss', alpha=0.8)

    plt.plot(np.arange(len(results.loc['val_gm_loss',:][0])), results.loc['val_gm_loss',:][0], 
                 label='Validation loss', alpha=0.8)

    plt.xlabel('Epoches', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title('Loss - simCLR', fontsize=20)
    plt.legend(fontsize=20)
    plt.show()

    save_folder = './figures/{}'.format(model_name)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    fig.savefig(os.path.join(save_folder, '{}_train_val_losses.pdf'.format(model_name)))
    
    plot_loss(results, 'val_gm_corr', save_folder=save_folder, fig_save_name = model_name, 
              title='Correlation - {}'.format(model_name), ylab='correlation')
    plot_loss(results, 'val_gm_p', save_folder=save_folder, fig_save_name = model_name, 
              title='p-value - {}'.format(model_name), ylab='pvalue')
    
    
import plotly
from plotly.offline import init_notebook_mode, iplot
from plotly.graph_objs import *
from plotly import tools

def scree_plot(pca, title, width=800, height=600):
    explained_var_ratio = pca.explained_variance_ratio_
    cum_explained_var = np.cumsum(explained_var_ratio)
    plotly.offline.init_notebook_mode()
    trace1 = Bar(y = explained_var_ratio, name = 'Variance ratio',
                text = "Explained variance")
    trace2 = Scatter(y = cum_explained_var,
                mode = "lines", name = 'cumulative',
                text = "Cumulative explained variance")
    layout = Layout(title = title, title_x=0.45, autosize=False,
                width=width, height=height,
              xaxis= dict(title= 'Principal components',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Explained variance ratio',tickformat=".2%", ticklen= 5,zeroline= False))
    plotly.offline.iplot({'data': [trace1, trace2], 'layout': layout})
    

def biplot_gm(score, coeff, arrow_width = .0015, labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.figure(figsize=(10,8))
    plt.scatter(xs * scalex,ys * scaley,s=5)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5, width = arrow_width)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), 
                     color = 'green', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], 
                     color = 'g', ha = 'center', va = 'center')
 
    plt.xlabel("PC{}".format(1), fontsize=20)
    plt.ylabel("PC{}".format(2), fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid()
    plt.show()
    
    
### for poster
from matplotlib import colors
import matplotlib

def scatter_2d_cate(dataset, score, pca, column_name, title, 
                    save_folder, save_file_name, cmap_name='Set1', fontsize=20, marker_size=60, transparency=1):
    marker_list = np.array(["o", "s", "v", "+", "x","*"])
    levels = np.unique(dataset[column_name], return_inverse=True)
    #plt.style.use('ggplot')
    plt.figure(figsize=(10, 10))
    c_n = len(levels[0])
    col_list = matplotlib.cm.get_cmap(cmap_name)(range(c_n))
    handles = []
    for i in range(c_n):
        ind = np.where(levels[1]==i)
        scatter = plt.scatter(score[ind,0], score[ind,1], color=col_list[levels[1][ind]], s=marker_size, 
                              alpha=transparency, marker=marker_list[i])
        handles.append(scatter)
    plt.title(title, fontsize=fontsize)
    legend = plt.legend(handles=handles, labels=list(levels[0]), title=column_name, fontsize=20)
    legend.get_title().set_fontsize('20')
    plt.xticks(fontsize=fontsize, rotation=45)
#     plt.xlabel('Variance Explained by PC1: {:.2f}%'.format(pca.explained_variance_ratio_[0]*100), fontsize=fontsize)
    plt.xlabel('PC1', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
#     plt.ylabel('Variance Explained by PC2: {:.2f}%'.format(pca.explained_variance_ratio_[1]*100), fontsize=fontsize)
    plt.ylabel('PC2', fontsize=fontsize)
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, '{}.pdf'.format(save_file_name))
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    

    
from matplotlib import colors
import matplotlib

def scatter_2d_cate_v2(dataset, score, pca, column_name, title, cmap_name='Set1', fontsize=20, marker_size=60, transparency=1):
    marker_list = np.array(["o", "s", "v", "+", "x","*"])
    levels = np.unique(dataset[column_name], return_inverse=True)
    #plt.style.use('ggplot')
    plt.figure(figsize=(10, 10))
    c_n = len(levels[0])
    col_list = matplotlib.cm.get_cmap(cmap_name)(range(c_n))
    handles = []
    for i in range(c_n):
        ind = np.where(levels[1]==i)
        scatter = plt.scatter(score[ind,0], score[ind,1], color=col_list[levels[1][ind]], s=marker_size, 
                              alpha=transparency, marker=marker_list[i])
        handles.append(scatter)
    plt.title('{} (by {})'.format(title,column_name), fontsize=fontsize)
    legend = plt.legend(handles=handles, labels=list(levels[0]), title=column_name, fontsize=fontsize)
    legend.get_title().set_fontsize('20')
    plt.xticks(fontsize=fontsize)
    plt.xlabel('Variance Explained by PC1: {:.2f}%'.format(pca.explained_variance_ratio_[0]*100), fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylabel('Variance Explained by PC2: {:.2f}%'.format(pca.explained_variance_ratio_[1]*100), fontsize=fontsize)
    plt.show()
    
    
### for poster
def scatter_2d_cont(dataset, score, pca, column_name, title, train_val, cmap_name='YlGnBu', fontsize=20, marker_size=100):
   # plt.style.use('ggplot')
    plt.figure(figsize=(12,10))
    scatter = plt.scatter(score[:,0], score[:,1], c=dataset[column_name], cmap=cmap_name, s=marker_size,
                          vmin=np.min(dataset[column_name]), vmax=np.max(dataset[column_name]))
    cbar = plt.colorbar(scatter)
    cbar.set_label(column_name)
    plt.title('{}'.format(title), fontsize=fontsize)
    plt.xticks(fontsize=fontsize, rotation=45)
#     plt.xlabel('Variance Explained by PC1: {:.2f}%'.format(pca.explained_variance_ratio_[0]*100), fontsize=fontsize)
    plt.xlabel('PC1', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
#     plt.ylabel('Variance Explained by PC2: {:.2f}%'.format(pca.explained_variance_ratio_[1]*100), fontsize=fontsize)
    plt.ylabel('PC2', fontsize=fontsize)
    plt.savefig('../poster/gemini-master/fig/MB_SupCon_{}_{}.pdf'.format(train_val, column_name), bbox_inches='tight')
    plt.show()
    
def scatter_2d_cont_v2(dataset, score, pca, column_name, title, cmap_name='YlGnBu', fontsize=20, marker_size=60):
   # plt.style.use('ggplot')
    plt.figure(figsize=(12,10))
    scatter = plt.scatter(score[:,0], score[:,1], c=dataset[column_name], cmap=cmap_name, s=marker_size,
                          vmin=np.min(dataset[column_name]), vmax=np.max(dataset[column_name]))
    cbar = plt.colorbar(scatter)
    cbar.set_label(column_name)
    plt.title('{} (by {})'.format(title,column_name), fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.xlabel('Variance Explained by PC1: {:.2f}%'.format(pca.explained_variance_ratio_[0]*100), fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylabel('Variance Explained by PC2: {:.2f}%'.format(pca.explained_variance_ratio_[1]*100), fontsize=fontsize)
    plt.show()
    
    
# SubjectID function
def plot_subject(dataset, score, title, levels_idx=None, cmap='RdBu', legend_col=3, bbox_to_anchor=(1, 1), fontsize=20):
    plt.figure(figsize=(10,10))
    levels = np.unique(dataset['SubjectID'], return_inverse=True)
    if levels_idx==None:
        enu_lvls = levels[0]
        norm_i = matplotlib.colors.Normalize(vmin=np.min(levels[1]), vmax=np.max(levels[1]))
    else:
        enu_lvls = levels[0][levels_idx]
        norm_i = matplotlib.colors.Normalize(vmin=0, vmax=len(levels_idx)-1)
    for i,lvl in enumerate(enu_lvls):
        idx = np.where(dataset['SubjectID']==lvl)
        clr = matplotlib.cm.get_cmap(cmap)(norm_i(i))
        plt.scatter(score[idx,0], score[idx,1], label=lvl, c=[clr])
    plt.title('{} (by {})'.format(title,'SubjectID'), fontsize=fontsize)
    legend = plt.legend(title='SubjectID',bbox_to_anchor=bbox_to_anchor, ncol=legend_col, fontsize=fontsize)
    legend.get_title().set_fontsize('20')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.show()
    
    
## define a function for plotting by different methods
def plot_by_score(dataset, score, pca, title, random_state, subj_num = 10, fontsize=20, transparency=1):
    
    np.random.seed(random_state)
    
    scatter_2d_cate(dataset, score, pca, 'IR_IS_classification', title, fontsize=fontsize, transparency=transparency)
    scatter_2d_cate(dataset, score, pca, 'Sex', title, fontsize=fontsize, transparency=transparency)
    scatter_2d_cate(dataset, score, pca, 'Race', title, fontsize=fontsize, transparency=transparency)
    
    scatter_2d_cont(dataset, score, pca, 'Age', title, fontsize=fontsize)
    scatter_2d_cont(dataset, score, pca, 'BMI', title, fontsize=fontsize)
    scatter_2d_cont(dataset, score, pca, 'SSPG', title, fontsize=fontsize)
    
    plot_subject(dataset, score, title, fontsize=fontsize)

    levels = np.unique(dataset['SubjectID'], return_inverse=True)
    lvls_idx = list(np.random.choice(range(np.max(levels[1])),subj_num))
    plot_subject(dataset, score, title, levels_idx=lvls_idx,
                 cmap="tab10", legend_col=1, bbox_to_anchor=(1.2, 1), fontsize=fontsize)
                    
    return lvls_idx

## define a function to draw LRP bar plots
def lrp_plot(covariate, feature_idx, R_0_bycol_sorted, x_label=False, title='Relevance Score'):
    fig, ax = plt.subplots(figsize=(20,8))
    ax.bar(range(len(R_0_bycol_sorted)),R_0_bycol_sorted)
    ax.set_xticks(range(len(R_0_bycol_sorted)))
    ax.tick_params(axis='x', labelrotation = 90, labelsize=10)
    ax.tick_params(axis='y', labelsize=15)
    if x_label:
        ax.set_xticklabels(len(R_0_bycol_sorted).index, rotation=30, ha='right')
    else:
        ax.set_xticklabels("", rotation=30, ha='right')
    ax.set_title('{}\nFeature {} ({})'.format(title, feature_idx, covariate), fontsize=40)
    save_folder='./LRP_plots/{}'.format(covariate)
    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(os.path.join(save_folder, 'LRP_feature_{}_{}.pdf'.format(feature_idx, covariate)), bbox_inches='tight')
    plt.show()

## define a function to draw LRP heatmap
def lrp_heatmap(covariate, feature_idx, R_0_df, title='Heatmap of LRP'):
    ### Reference: https://stackoverflow.com/questions/27924813/extracting-clusters-from-seaborn-clustermap
    import seaborn as sns
    from scipy.spatial import distance
    from scipy.cluster import hierarchy

    row_linkage = hierarchy.linkage(
        distance.pdist(R_0_df))

    col_linkage = hierarchy.linkage(
        distance.pdist(R_0_df.T))

    plt.figure(figsize=(100,10))
    sns.set_style(style='white')
    heatmap = sns.clustermap(R_0_df, row_linkage=row_linkage, col_linkage=col_linkage,
                             cmap="coolwarm", vmin=-0.1, vmax=0.1,
                             cbar_pos=(.02, .5, .03, .2))
    heatmap.fig.suptitle('{}\n({} - Feature {})'.format(title, covariate, feature_idx), y=1.05, fontsize=20) 
    plt.title('Relevance Score')
    save_folder='./LRP_plots/{}'.format(covariate)
    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(os.path.join(save_folder, 'LRP_heatmap_{}_{}.pdf'.format(feature_idx, covariate)), bbox_inches='tight')
    plt.show()