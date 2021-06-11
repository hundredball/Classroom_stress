from os import path

import numpy as np
import pandas as pd
import scipy.io as sio

def find_common_channels():
    '''
    Find common channels across all samples
    '''
    channels = sio.loadmat('./data/ch_lib.mat')
    
    channel_all_list = [channels['ch_lib'][0][0][0][i_channel][0] for i_channel in range(30)]

    for i_sub in range(len(channels['ch_lib'])):
    
        if len(channels['ch_lib'][i_sub][0][0])==0:
            continue
        
        channel_list = [channels['ch_lib'][i_sub][0][0][i_channel][0] for i_channel in range(len(channels['ch_lib'][i_sub][0][0]))]
        for channel in channel_all_list:
            if channel not in channel_list:
                channel_all_list.remove(channel)

    return channel_all_list

def read_data():
    
    '''
    Read EEG data and stress level
    -----
    
    Returns
    --------
    EEG_list: list of 2d array
        i: example
        j: channel
        k: time
    labels: 1d array
        stress level of data
    subjects: 1d array
        subject ID of data
    '''
    num_samples = 171
    
    # --- Prepare predictors ---
    print('Load data from .mat files...')
    channels = sio.loadmat('./data/ch_lib.mat')
    channels_common = find_common_channels()
    
    EEG_list = []
    miss_sample = []
    subjects = []

    for i_sample in range(num_samples):
        file_path = './data/rawdata/%d.mat'%(i_sample+1)
        
        # Continue if file of sample doesn't exist
        if not path.exists(file_path):
            miss_sample.append(i_sample)
            continue
            
        #print('Sub ', i_sample+1)
        EEG = sio.loadmat(file_path)
        EEG = EEG['tmp']
        
        # Select common channels
        channels_i = channels['ch_lib'][i_sample][0][0]
        channels_select_i = [i for i in range(len(channels_i)) if channels_i[i][0] in channels_common]
        EEG_list.append(EEG[channels_select_i,:])
    
    # --- Prepare labels --- 
    df = pd.read_csv('./data/SFC_DASS21.csv')
    
    # Get increased stress or normal label for each subject
    num_sub = 26
    group_mean = df.groupby(by='subject').mean()
    labels = []

    for i in range(len(df)):
        if i in miss_sample:
            continue
        
        stress_sub = df.iloc[i]['stress']
        sub = df.iloc[i]['subject']
        threshold = group_mean.loc[sub]['stress']
    
        labels.append(stress_sub>threshold)
        subjects.append(sub)
    
    labels = np.asarray(labels, 'int')
    subjects = np.asarray(subjects, 'int')
    
    return EEG_list, labels, subjects