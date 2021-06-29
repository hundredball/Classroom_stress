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
        1: DASS increase or normal within each subject (>= mean+std)
        2: DASS increase or normal of DASS standard
        3: DSS increase or normal within each subject (>= mean+std)
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
    channels_common = find_common_channels()
    max_num_channel = len(channels_common) if common_flag else 30

    # Read dataframes of DASS
    df_DASS = pd.read_csv('./data/SFC_DASS21.csv')
    df_DASS.loc[:,'session'] = [date.replace("'",'') for date in df_DASS['session']] # Remove ' in session
    df_DASS.loc[:,'DASS_record_number'] = list(range(1,len(df_DASS)+1))
    group_DASS = df_DASS.groupby(by='subject')
    mean_DASS = group_DASS.mean()
    std_DASS = group_DASS.std()

    # Add corresponding channels in DASS dataframe
    channels_list = []
    for i_sample in range(len(df_DASS)):
    
        # Select channels
        channels_i = channels['ch_lib'][i_sample][0][0]
        channels_i_name = [channels_i[i][0] for i in range(len(channels_i))]

        channels_list.append(channels_i_name)
    df_DASS.loc[:,'channels'] = channels_list

    # Read dataframes of DSS
    subjects = list(range(1,27))
    df_DSS = pd.DataFrame()

    for subject in subjects:
        if subject <= 18:
            file_path = './data/DSS/first semester/%s/first_datebystress%d.txt'%(str(subject).zfill(2), subject)
        else:
            file_path = './data/DSS/second semester/%s/second_datebystress%d.txt'%(str(subject-18).zfill(2), subject-18)

        df = pd.read_csv(file_path, sep = '\s+', header=0, names=['session','stress_DSS'], index_col=False)
        df.loc[:,'subject'] = [subject]*len(df)
        df_DSS = df_DSS.append(df)

    df_DSS = df_DSS.reset_index(drop=True)
    group_DSS = df_DSS.groupby(by='subject')
    mean_DSS = group_DSS.mean()
    std_DSS = group_DSS.std()
    # Take average of stress if some subjects have DSS record on the same date
    df_DSS = df_DSS.groupby(by=['subject', 'session']).mean().reset_index()

    # Merge dataframes of DASS and recording system
    df_summary = pd.read_csv('./data/summary_NCTU_RWN-SFC.csv')
    df_summary = df_summary.dropna(how='all')
    df_summary = df_summary.rename(columns = {'folder/session':'session', 'labelID':'subject'})
    df_summary.loc[:,'subject'] = [int(df_summary['subject'].values[i][1:]) for i in range(len(df_summary['subject']))]
    df_all = pd.merge(df_DASS, df_summary, how='inner', sort=False, on=['subject','session'])

    # Change session, e.g. 20150506_s26 -> 2015-0506
    df_all.loc[:,'session'] = [date.split('_')[0][:4] + '-' + date.split('_')[0][4:] for date in df_all['session']]

    # Merge dataframes of DSS and df_all
    df_all = pd.merge(df_all, df_DSS, how='inner', sort=False, on=['subject','session'])

    labels = []
    EEG_list = []
    drop_list = []
    channels_list = []

    for i_sample in range(len(df_all)):
        file_path = './data/rawdata/%d.mat'%(df_all.loc[i_sample,'DASS_record_number'])

        # Continue if file of sample doesn't exist
        if not path.exists(file_path):
            drop_list.append(i_sample)
            continue

        # Get increased stress or normal label for each subject
        sub = df_all.loc[i_sample]['subject']
        stress_DASS = df_all.loc[i_sample]['stress']
        stress_DSS = df_all.loc[i_sample]['stress_DSS']

        if label_format==1:
            threshold = mean_DASS.loc[sub]['stress'] + std_DASS.loc[sub]['stress']
            labels.append(stress_DASS>threshold)
        elif label_format==2:
            labels.append(stress_DASS>15)
        elif label_format==3:
            threshold = mean_DSS.loc[sub]['stress_DSS'] + std_DSS.loc[sub]['stress_DSS']
            labels.append(stress_DSS>threshold)

        # Load EEG
        EEG = sio.loadmat(file_path)
        EEG = EEG['tmp']

        # Select channels
        channels_i = df_all.iloc[i_sample]['channels']
        channels_i_index = [i for i in range(len(channels_i)) if (channels_i[i] in channels_common or not common_flag)]
        channels_i_name = [channels_i[i] for i in range(len(channels_i)) if (channels_i[i] in channels_common or not common_flag)]

        channels_list.append(channels_i_name)
        EEG_list.append(EEG[channels_i_index,:])

    df_all = df_all.drop(drop_list)
    df_all = df_all.reset_index(drop=True)

    if common_flag:
        df_all['channels'] = channels_list

    labels = np.asarray(labels, 'int')
    
    return EEG_list, labels, df_all
    