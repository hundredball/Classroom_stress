import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

import dataloader

def remove_trials(EEG_list, labels, df_all, remove_indices = None):
    '''
    Remove trials of given indices
    '''
    
    # Bad trials by visual inspection
    if remove_indices == None:
        remove_indices = {96, 97, 101, 109, 111, 113, 115, 116, 119, 141, 147, 164}
    
    select_indices = [i for i in range(len(EEG_list)) if i not in remove_indices]
    
    EEG_list = [EEG_list[i] for i in range(len(EEG_list)) if i in select_indices]
    labels = labels[select_indices]
    df_all = df_all.iloc[select_indices,:].reset_index(drop=True)
    
    return EEG_list, labels, df_all

def select_features(X_train, X_test, Y_train):
    
    assert isinstance(X_train, np.ndarray) and X_train.ndim==2
    
    sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear'))
    sel_.fit(X_train, Y_train)
    indices = np.where(sel_.estimator_.coef_!=0)[1]
    
    return X_train[:, indices], X_test[:, indices]
    

def avg_channels_into_regions(EEG_list, df_all):
    '''
    Average over channels by six regions
    
    Parameter
    ---------
    EEG_list : list of 2d array
        i : example
        j : channel
        k : time sample
    
    df_all : DataFrame
        contains channel information
    
    Return
    ------
    EEG_regions_list : list of 2d array
        i : example
        j : region
        k : time sample
    '''
    
    EEG_regions_list = []

    for i in range(len(EEG_list)):
        EEG_regions = np.zeros((6, EEG_list[i].shape[1]))
        indices_channels = [[] for i in range(6)]
    
        for j in range(EEG_list[i].shape[0]):
            channel = df_all.loc[i, 'channels'][j]
        
            if channel[0] == 'F':
                indices_channels[0].append(j)
            elif channel[0] == 'C':
                indices_channels[2].append(j)
            elif channel[0] == 'P' or channel in ['T5','T6']:
                indices_channels[4].append(j)
            elif channel[0] == 'O':
                indices_channels[5].append(j)
            elif channel[0] == 'T' and int(channel[-1])%2 == 1:
                indices_channels[1].append(j)
            else:
                indices_channels[3].append(j)
    
        for i_region in range(6):
            EEG_regions[i_region,:] = np.mean(EEG_list[i][indices_channels[i_region],:], axis=0)
        
        EEG_regions_list.append(EEG_regions)
        
    # Replace channels with regions
    df_all['channels'] = [['F','LT','C','RT','P','O'] for _ in range(len(df_all))]
        
    return EEG_regions_list