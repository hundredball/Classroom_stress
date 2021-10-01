from os import path

import numpy as np
import pandas as pd
import scipy.io as sio
import glob

def find_common_channels(channels):
    '''
    Find common channels across all samples
    '''
    #channels = sio.loadmat('./data/ch_lib.mat')
    
    channel_all_list = [channels[0][0][0][i_channel][0] for i_channel in range(30)]

    for i_sub in range(len(channels)):
    
        if len(channels[i_sub][0][0])==0:
            continue
        
        channel_list = [channels[i_sub][0][0][i_channel][0] for i_channel in range(len(channels[i_sub][0][0]))]
        for channel in channel_all_list:
            if channel not in channel_list:
                channel_all_list.remove(channel)

    return channel_all_list

def read_DASS(fileName):
    '''
    Read from csv file, return dataframe
    '''
    df_DASS = pd.read_csv(fileName)
    df_DASS.loc[:,'session'] = [date.replace("'",'') for date in df_DASS['session']] # Remove ' in session
    df_DASS['session'] = df_DASS['session'].str[:8]
    
    return df_DASS

def read_channels(fileName): 
    '''
    Read from mat file, return List[List[str]]
    '''
    channels = sio.loadmat(fileName)['ch_lib']
    rest = fileName.find('rest')!=-1
    num_sample = len(channels[0]) if rest else len(channels)
    
    channels_list = []
    for i_sample in range(num_sample):
    
        # Select channels
        channels_i = channels[0][i_sample][0] if rest else channels[i_sample][0][0]
        channels_i_name = [channels_i[i][0] for i in range(len(channels_i))]

        channels_list.append(channels_i_name)
        
    return channels_list

def read_data(label_format=1, data_folder = 'rawdata'):
    '''
    Read EEG data and stress level
    -----
    label_format: int
        1: DASS increase or normal within each subject (>= mean+std)
        2: DASS increase or normal of DASS standard
        3: DSS increase or normal within each subject (>= mean+std)
        4: Compare before and after exam DASS increase and decrease (responder) or DASS static (non-responder), only load before EEG
        5: Same as 4, but only load after EEG
        6: Same as 4, but load before and after EEG
        
    data_folder : str
        rawdata, preprocessed, rest
    
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
    
    if label_format in [4,5,6]:
        assert data_folder == 'rest'
     
    print('Load data from .mat files...')

    if data_folder == 'rawdata':
        channels = read_channels('./data/ch_lib.mat')
        EEG_fieldName = 'tmp'
    elif data_folder == 'preprocessed':
        channels = read_channels('./data/ch_lib.mat')
        EEG_fieldName = 'data'
    elif data_folder == 'rest':
        channels = read_channels('./data/rest/ch_lib.mat')
        channels = [channels[i//2] for i in range(2*len(channels))]
        files = glob.glob('./data/rest/2015*')
        sessions, subjectIDs, fileNames, periods = [], [], [], []
        for file in files:
            sessions.append(file[12:20])
            subjectIDs.append(int(file[22:24]))
            fileNames.append(file[12:])
            periods.append('before' if file[-5]=='b' else 'after')
        df_trials = pd.DataFrame({'session': sessions, 'subject': subjectIDs, 'fileName': fileNames, 'period':periods})
        EEG_fieldName = 'rest_1_data'    # Replace 1 by a or b according to data
    else:
        raise ValueError

    # Read dataframes of DASS
    if data_folder == 'rest':
        df_DASS_before = read_DASS('./data/SFC_DASS21_before stress.csv')
        df_DASS_after = read_DASS('./data/SFC_DASS21_after stress.csv')
        df_DASS_before['period'] = ['before']*len(df_DASS_before)
        df_DASS_after['period'] = ['after']*len(df_DASS_after)
        df_DASS = pd.concat([df_DASS_before, df_DASS_after], ignore_index=True)
        df_DASS = pd.merge(df_DASS, df_trials, how='inner', on=['subject', 'session', 'period'])
        df_DASS = df_DASS.sort_values(by=['session', 'subject', 'period']).reset_index(drop=True)
    else:
        df_DASS = read_DASS('./data/SFC_DASS21.csv')
        df_DASS.loc[:,'fileName'] = ['%d.mat'%(i) for i in range(1, len(df_DASS)+1)]
    group_DASS = df_DASS.groupby(by='subject')
    mean_DASS = group_DASS.mean()
    std_DASS = group_DASS.std()
    df_DASS['channels'] = channels
    
    # Merge dataframes of DASS and recording system
    df_summary = pd.read_csv('./data/summary_NCTU_RWN-SFC.csv')
    df_summary = df_summary.dropna(how='all')
    df_summary = df_summary.rename(columns = {'folder/session':'session', 'labelID':'subject'})
    df_summary.loc[:,'subject'] = [int(df_summary['subject'].values[i][1:]) for i in range(len(df_summary['subject']))]
    df_summary['session'] = df_summary['session'].str[:8]
    df_summary = df_summary[['session','subject','channelLocations']]
    df_all = pd.merge(df_DASS, df_summary, how='left', sort=False, on=['subject','session'])
    
    # Read dataframes of DSS
    if label_format == 3:
        subjects = list(range(19,27)) if data_folder=='rest' else list(range(1,27)) 
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

        # Change session, e.g. 20150506_s26 -> 2015-0506
        df_all.loc[:,'session'] = [date.split('_')[0][:4] + '-' + date.split('_')[0][4:] for date in df_all['session']]

        # Merge dataframes of DSS and df_all
        df_all = pd.merge(df_all, df_DSS, how='inner', sort=False, on=['subject','session'])

    labels = []
    EEG_list = []
    drop_list = []
    channels_list = []

    for i_sample in range(len(df_all)):
        file_path = './data/%s/%s'%(data_folder,df_all.loc[i_sample,'fileName'])

        # Continue if file of sample doesn't exist
        if not path.exists(file_path):
            drop_list.append(i_sample)
            continue

        # Get increased stress or normal label for each subject
        sub = df_all.loc[i_sample]['subject']
        stress_DASS = df_all.loc[i_sample]['stress']
        
        if label_format==1:
            threshold = mean_DASS.loc[sub]['stress'] + std_DASS.loc[sub]['stress']
            labels.append(stress_DASS>threshold)
        elif label_format==2:
            labels.append(stress_DASS>15)
        elif label_format==3:
            stress_DSS = df_all.loc[i_sample]['stress_DSS']
            threshold = mean_DSS.loc[sub]['stress_DSS'] + std_DSS.loc[sub]['stress_DSS']
            labels.append(stress_DSS>threshold)
        elif label_format in [4,5,6] and i_sample<len(df_all)//2:
            labels.append(int(df_all.loc[i_sample*2, 'stress']!=df_all.loc[i_sample*2+1, 'stress']))

        # Load EEG
        EEG = sio.loadmat(file_path)
        if data_folder == 'rest':
            rest_period = 'a' if df_all.loc[i_sample, 'period']=='after' else 'b'
            EEG = EEG[EEG_fieldName.replace('1', rest_period)]
        else:
            EEG = EEG[EEG_fieldName]

        if label_format not in [4,5] or (label_format==4 and rest_period=='b') or (label_format==5 and rest_period=='a'):
            EEG_list.append(EEG)
            
        # Drop after period if label format is 4
        if (label_format==4 and rest_period=='a') or (label_format==5 and rest_period=='b'):
            drop_list.append(i_sample)

    df_all = df_all.drop(drop_list)
    df_all = df_all.reset_index(drop=True)

    labels = np.asarray(labels, 'int')
    
    return EEG_list, labels, df_all
    