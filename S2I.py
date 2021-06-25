#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 2021

@author: hundredball
"""

import os
import time

import numpy as np
import pandas as pd
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.model_selection import train_test_split, KFold
import argparse
import pickle
from PIL import Image

import dataloader
import bandpower

parser = argparse.ArgumentParser(description='Signal to image')
parser.add_argument('-m', '--mode', default='normal', type=str, help='Generating method')
parser.add_argument('-d', '--data_cate', default=1, type=int, help='Category of data')
parser.add_argument('-t', '--num_time', default=1, type=int, help='Number of frames for each example')
parser.add_argument('-r', '--remove_threshold', default=60.0, type=float, help='SL threshold for removing trials')
parser.add_argument('-f', '--num_fold', default=1, type=int, help='Number of folds of cross validation')
parser.add_argument('-s', '--num_split', default=1, type=int, help='If >1, for ensemble methods')
parser.add_argument('--split_mode', default=3, type=int, help='Mode for spliting training data')
parser.add_argument('--num_channels', default=21, type=int, help='Number of channels of loaded dataset')
parser.add_argument('--subject_ID', default=100, type=int, help='Subject ID, 100 for all subjects')
parser.add_argument('--normal_sub', action='store_true', help='Normalize subjects by mean power')

def fig2data ( fig ):
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf = buf.reshape((w,h,4))
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def fig2img ( fig ):
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )

def generate_topo(bandpower, info_df):
    '''
    Generate mixed topoplot

    Parameters
    ----------
    bandpower: list of 2d array
        (example, channel, band)
    info_df : pandas DataFrame
        subject ID and channels
    
    Returns
    -------
    fileNames : numpy 1d array (epoch)
        file names of mixed images

    '''
    assert hasattr(powers,'__iter__') and all([power.ndim==2 for power in powers])
    assert isinstance(info_df, pd.DataFrame)
    
    start_time = time.time()
    print('[%f] Generating topoplots...'%(time.time()-start_time))
    
    # Get recording system
    record_info = pd.read_csv('./data/summary_NCTU_RWN-SFC.csv')
    record_info = record_info.dropna(how = 'all')
    systems = record_info['channelLocations'].unique()
    
    # ----- Edit here -----
    
    # Get channel info for two systems
    channel_info = {}
    channel_info[systems[0]] = pd.read_csv('./data/30ch_loc_NuAmps.csv')
    channel_info[systems[1]] = pd.read_csv('./data/30ch_loc_SynAmps2.csv')
    
    # Change coordinate from 0 toward naison to 0 toward right ear
    channel_info[:,2] = 90-channel_info[:,2]
    channel_info[systems[0]]['theta'] = 90 - channel_info[systems[0]]['theta']
    channel_info[systems[1]]['theta'] = 90 - channel_info[systems[1]]['theta']

    band_name = ['theta', 'alpha', 'beta']
    cmap_name = ['Reds', 'Greens', 'Blues']

    # Turn interactive plotting off
    plt.ioff()

    # Topoplot
    for i_data in range(num_example):
        min_ = {}
        scale = {}
        im = {}
        subID = info_df

        # Plot topo for each band
        for i_band in range(3):
            fig, ax = plt.subplots(figsize=(4,4))
            ax.axis('off')
            #fig = plt.figure()
            #ax = fig.add_subplot(111, aspect=1)
            n_angles = 48
            n_radii = 200
            radius = 1.0
            radii = np.linspace(0, radius, n_radii)
            angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

            # Calculate channel locations on the plot
            scale_radius = radius/0.5
            scale_arc = scale_radius*channel_info[:,1]
            plot_loc = np.zeros((num_channels, 2)) # first for x, second for y
            plot_loc[:,0] = scale_arc*np.cos(np.array(channel_info[:,2]*np.pi/180, dtype = np.float))
            plot_loc[:,1] = scale_arc*np.sin(np.array(channel_info[:,2]*np.pi/180, dtype = np.float))
            
            # Set pixel values 0~1
            min_[i_band] = np.min(bandpower[i_data,:,i_band])
            channel_values = bandpower[i_data,:,i_band]-min_[i_band]
            scale[i_band] = np.max(channel_values)
            channel_values = channel_values/scale[i_band]
            
            # Add couple of min values to outline for interpolation
            add_x = np.reshape(radius*np.cos(angles), (len(angles), 1))
            add_y = np.reshape(radius*np.sin(angles), (len(angles), 1))
            add_element = np.concatenate((add_x, add_y), axis=1)
            plot_loc = np.concatenate((plot_loc, add_element), axis=0)
            channel_values = np.concatenate((channel_values, np.zeros(len(angles))))

            # Interpolate 
            angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1) 
            x = (radii * np.cos(angles)).flatten()
            y = (radii * np.sin(angles)).flatten()
            z = griddata(plot_loc, channel_values, (x, y), method = 'cubic', fill_value=0, rescale=True)

            triang = tri.Triangulation(x, y)
            ax.set_xlim((-1, 1))
            ax.set_ylim((-1, 1))
            tcf = ax.tricontourf(triang, z, cmap = cmap_name[i_band], levels=60)   # Reds, Greens, Blues
            #fig.colorbar(tcf)
            im[band_name[i_band]] = fig2img(fig)
            
            plt.close()
        
        for i_band, band in enumerate(band_name):
            im[band] = np.array(im[band])
            # Scale back rgb
            #im[band] = im[band]*scale[i_band]+min_[i_band]
            
        figure_mix = im['theta'] + im['alpha'] + im['beta']
        figure_mix = figure_mix[:,:,:3]
            
        #figure_mix[:,:,3] = 255
        # Change to uint8
        figure_mix = np.asarray(figure_mix, dtype=np.uint8)
        
        if index_split == -1:
            dirName = 'test'
        else:
            dirName = 'train%d'%(index_split)
        
        fileName = '%d_mix_%d'%(i_data, i_time)
        plt.imsave('./images/exp%d/%s/%s.png'%(index_exp, dirName, fileName), figure_mix)

        # Only save fileNames for the first time step
        if i_time == 0:
            fileNames[i_data] = '%s/%s'%(dirName, fileName)
        
    print('[%f] Finished all topoplots!'%(time.time()-start_time))
    
        
    return fileNames
    
def split(fileNames, SLs, test_size=0.1, random=True, index_exp=0):
    '''
    Split training and testing set by creating csv files for referencing

    Parameters
    ----------
    fileNames : numpy 1d array (epoch)
        File names of mixed images
    SLs : numpy 1d array (epoch)
        Solution latency of those trials
    test_ratio : float, optional
        Ration of testing set. The default is 0.1.
    random : bool, optional
        The training or testing data in fileNames are random or not. The default is True.
        
    Returns
    -------
    None.

    '''
    assert isinstance(fileNames, np.ndarray) and fileNames.ndim == 1
    assert isinstance(SLs, np.ndarray) and SLs.ndim == 1
    assert isinstance(random, bool)
    assert (random and isinstance(test_size, float) and 0<=test_size<=1) or\
        ((not random) and isinstance(test_size, int) and 0<=test_size<=fileNames.shape[0])
    assert fileNames.shape[0] == SLs.shape[0]
    assert isinstance(index_exp, int)
    
    # Split for training and testing data
    if not random:
        X_train, X_test = fileNames[:-test_size], fileNames[-test_size:]
        Y_train, Y_test = SLs[:-test_size], SLs[-test_size:]
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(fileNames, SLs, test_size=test_size, random_state=42)
    
    # Save csv for dataloader
    X_train_df = pd.DataFrame({'fileName':X_train})
    X_train_df.to_csv('./images/exp%d/train0_img.csv'%(index_exp))
    
    X_test_df = pd.DataFrame({'fileName':X_test})
    X_test_df.to_csv('./images/exp%d/test_img.csv'%(index_exp))
    
    Y_train_df = pd.DataFrame({'solution_time':Y_train})
    Y_train_df.to_csv('./images/exp%d/train0_label.csv'%(index_exp))
    
    Y_test_df = pd.DataFrame({'solution_time':Y_test})
    Y_test_df.to_csv('./images/exp%d/test_label.csv'%(index_exp))
    
    print('Generate files for dataset referencing')
    
def generate_csv(fileNames, SLs, index_exp, index_split):
    '''
    Genereate csv files of img and label for ensemble methods

    Parameters
    ----------
    fileNames : numpy 1d array
        File names of the topoplots (including dirName and fileName)
    SLs : numpy 1d array
        Solution latency of the topoplots
    index_exp : int
        Index of experiment for cross validation
    index_split : int
        Index of split cluster for ensemble methods
        -1 for testing set, 100 for all training set

    Returns
    -------
    None.

    '''
    assert isinstance(fileNames, np.ndarray) and fileNames.ndim==1
    assert isinstance(SLs, np.ndarray) and SLs.ndim==1
    assert isinstance(index_exp, int) and index_exp>=0
    assert isinstance(index_split, int)
    
    X_df = pd.DataFrame({'fileName':fileNames})
    Y_df = pd.DataFrame({'solution_time':SLs})
    if index_split != -1:
        X_df.to_csv('./images/exp%d/train%d_img.csv'%(index_exp, index_split))
        Y_df.to_csv('./images/exp%d/train%d_label.csv'%(index_exp, index_split))
    else:
        X_df.to_csv('./images/exp%d/test_img.csv'%(index_exp))
        Y_df.to_csv('./images/exp%d/test_label.csv'%(index_exp))
        
    print('Generated files for dataset referencing')
    
    
def S2I_main(powers, labels, info_df):
    '''
    Generate topoplot and csv

    Parameters
    ----------
    power: list of 2d array
        (example, channel, band)
    labels: numpy 1d array
        stress level
    info_df : Dataframe
        subject ID and channels

    Returns
    -------
    None.

    '''
    assert hasattr(powers,'__iter__') and all([power.ndim==2 for power in powers])
    assert isinstance(labels, np.ndarray) and labels.ndim==1
    assert isinstance(subjectIDs, pd.DataFrame)
    
    ## Edit here ##
    
    # Create folder for exp and train, test
    if not os.path.exists('./images/exp%d'%(index_exp)):
        os.makedirs('./images/exp%d'%(index_exp))
        os.makedirs('./images/exp%d/train0'%(index_exp))
        os.makedirs('./images/exp%d/test'%(index_exp))
    
    start_time = time.time()
    print('[%.1f] Signal to image (%s)'%(time.time()-start_time, mode))
    
    fileNames = generate_topo(powers, info_df)
    split(fileNames, labels, info_df)
    
    print('[%.1f] Finished S2I'%(time.time()-start_time))

if __name__ == '__main__':
    global args
    args = parser.parse_args()
    
    # Create folder for saving those images
    if not os.path.exists('./images'):
        os.makedirs('./images')
        
    # Load data
    signal, labels, info_df = dataloder.read_data(label_format=1)
    powers = bandpower.get_bandpower(signal, 500, low=[3,7,13], high=[7,13,30])
    
    # Generate topo and csv
    S2I_main(powers, freqs, labels, info_df)
    