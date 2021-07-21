import numpy as np
import matplotlib.pyplot as plt

import dataloader

def average_channels(EEG_list, df_all):
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
        
    return EEG_regions_list