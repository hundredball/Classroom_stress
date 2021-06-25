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

def read_data(label_format=1, common_flag = False):
    
    '''
    Read EEG data and stress level
    -----
    label_format: int
        1: increase or normal within each subject
        2: increase or normal of DASS standard
    common_flag : bool
        select common channels
    
    Returns
    --------
    EEG_list: list of 2d array
        i: example
        j: channel
        k: time
    labels: 1d array
        stress level of data
    df_all : Dataframe
        subject ID and channels
    '''
    
    print('Load data from .mat files...')
    channels = sio.loadmat('./data/ch_lib.mat')
    num_samples = len(channels['ch_lib'])
    sub_df = pd.read_csv('./data/SFC_DASS21.csv')
    group_mean = sub_df.groupby(by='subject').mean()
    channels_common = find_common_channels()
    max_num_channel = len(channels_common) if common_flag else 30
    
    # Merge dataframes of DASS and recording system
    df_summary = pd.read_csv('./data/summary_NCTU_RWN-SFC.csv')
    df_summary = df_summary.dropna(how='all')
    df_summary = df_summary.rename(columns = {'folder/session':'session', 'labelID':'subject'})
    df_summary['subject'] = [str(int(df_summary['subject'].values[i][1:])) for i in range(len(df_summary['subject']))]
    df_all = pd.concat([sub_df, df_summary], join='inner', axis=1)
    
    labels = []
    EEG_list = []
    info_df = pd.DataFrame(columns = ['channels'])

    for i_sample in range(num_samples):
        file_path = './data/rawdata/%d.mat'%(i_sample+1)
        
        # Continue if file of sample doesn't exist
        if not path.exists(file_path):
            df_all = df_all.drop(i_sample)
            continue
            
        # Get increased stress or normal label for each subject
        sub = sub_df.iloc[i_sample]['subject']
        
        stress_sub = sub_df.iloc[i_sample]['stress']
        if label_format==1:
            threshold = group_mean.loc[sub]['stress']
            labels.append(stress_sub>threshold)
        elif label_format==2:
            labels.append(stress_sub>15)
            
        # Load EEG
        EEG = sio.loadmat(file_path)
        EEG = EEG['tmp']
        
        # Select channels
        channels_i = channels['ch_lib'][i_sample][0][0]
        channels_i_index = [i for i in range(len(channels_i)) if (channels_i[i][0] in channels_common or not common_flag)]
        channels_i_name = [channels_i[i][0] for i in range(len(channels_i)) if (channels_i[i][0] in channels_common or not common_flag)]
        
        num_channel_i = len(channels_i_index)
        info_df.loc[i_sample] = [list(channels_i_name)]
        
        EEG_list.append(EEG[channels_i_index,:])
    
    df_all = df_all.reset_index(drop=True)
    info_df = info_df.reset_index(drop=True)
    
    df_all = pd.concat([df_all, info_df], axis=1)
    
    labels = np.asarray(labels, 'int')
    
    return EEG_list, labels, df_all