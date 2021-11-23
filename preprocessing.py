import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import scipy

import dataloader

fs = 1000

class StressData:
    
    def __init__(self, EEG_list, labels, df_all):
        '''
        EEG_list: list of 2D array
            i: trial
            j: channel
            k: time
        labels: 1D array
            labels regarding stress level
        df_all: DataFrame
            information for each trial (channels, stress, file name...)
        
        '''
        self.EEG_list = EEG_list
        self.labels = labels
        self.df_all = df_all
        
        # Remove trials that have mismatch number of channels in EEG_list and df_all
        remove_indices = set()
        for i in range(len(self.EEG_list)):
            if self.EEG_list[i].shape[0] != len(self.df_all.loc[i,'channels']):
                remove_indices.add(i)
        if remove_indices:        
            self.remove_trials(remove_indices)
        
    def remove_trials(self, remove_indices = None):
        '''
        Remove trials of given indices
        '''

        # Bad trials by visual inspection
        if remove_indices == None:
            remove_indices = {96, 97, 101, 109, 111, 113, 115, 116, 119, 141, 147, 164}
        print('Remove trials:')
        for index in remove_indices:
            print('Number: %d, Session: %s, Subject: %d'%(
                self.df_all.loc[index,'number'], self.df_all.loc[index,'session'], self.df_all.loc[index,'subject']))

        select_indices = [i for i in range(len(self.EEG_list)) if i not in remove_indices]

        self.EEG_list = [self.EEG_list[i] for i in range(len(self.EEG_list)) if i in select_indices]
        self.labels = self.labels[select_indices] 
        self.df_all = self.df_all.iloc[select_indices,:].reset_index(drop=True)

    def reReference(self, reference):
        '''
        Rereference data to given channel
        '''

        drop_index = set()
        for i in range(len(self.EEG_list)):

            try: 
                index_reference = self.df_all.loc[i, 'channels'].index(reference)
                self.EEG_list[i] = self.EEG_list[i] - self.EEG_list[i][index_reference,:]

            except ValueError:
                # If CZ is not in the channels, drop it
                drop_index.add(i)

        # Drop samples without CZ channels
        self.remove_trials(drop_index)

    def avg_channels_into_regions(self, mode=1):
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

        mode : int 
            1 : Average into 6 regions (frontal, left temporal, central, right temporal, parietal, occipital)
            2 : Ignore channels on the central line and average into 9 regions 
                (left frontal, right frontal, left temporal, left central, right central, right temporal, 
                 left parietal, right parietal, occipital)

        Return
        ------
        EEG_regions_list : list of 2d array
            i : example
            j : region
            k : time sample
        '''

        assert mode in [1,2]

        if mode == 1:
            regions = ['F','LT','C','RT','P','O']
        elif mode == 2:
            regions = ['RF','LF','RT','LT','RC','LC','RP','LP','O']

        EEG_regions_list = []

        for i in range(len(self.EEG_list)):
            EEG_regions = np.zeros((len(regions), self.EEG_list[i].shape[1]))
            indices_channels = [[] for i in range(len(regions))]

            for j in range(self.EEG_list[i].shape[0]):
                channel = self.df_all.loc[i, 'channels'][j]

                if mode == 1:
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
                elif mode == 2 and channel[-1].lower() != 'z':
                    if channel[0] == 'F' and int(channel[-1])%2 == 1:
                        indices_channels[1].append(j)
                    elif channel[0] == 'F' and int(channel[-1])%2 == 0:
                        indices_channels[0].append(j)
                    elif channel[0] == 'C' and int(channel[-1])%2 == 1:
                        indices_channels[5].append(j)
                    elif channel[0] == 'C' and int(channel[-1])%2 == 0:
                        indices_channels[4].append(j)
                    elif (channel[0] == 'P' and int(channel[-1])%2 == 1) or channel=='T5':
                        indices_channels[7].append(j)
                    elif (channel[0] == 'P' and int(channel[-1])%2 == 0) or channel=='T6':
                        indices_channels[6].append(j)
                    elif channel[0] == 'O':
                        indices_channels[8].append(j)
                    elif channel[0] == 'T' and int(channel[-1])%2 == 1:
                        indices_channels[3].append(j)
                    else:
                        indices_channels[2].append(j)

            for i_region in range(len(regions)):
                EEG_regions[i_region,:] = np.mean(self.EEG_list[i][indices_channels[i_region],:], axis=0)

            EEG_regions_list.append(EEG_regions)

        # Replace channels with regions
        self.df_all['channels'] = [regions for _ in range(len(self.df_all))]
        
        self.EEG_list = EEG_regions_list
        
        # Remove trials with nan
        indices = []
        for i in range(len(self.EEG_list)):
            if np.sum(np.isnan(self.EEG_list[i]))>0:
                indices.append(i)
                
        self.remove_trials(indices)
        
    def get_coherence(self, low, high):
        '''
        Add coherence of signal
        
        Parameter
        ----------
        low: list of numbers
            Low frequency of frequency bands
        high: list of numbers
            High frequency of freqeuncy bands
            
        Return
        ----------
        coherence: numpy 2d array
            Magnitude squared coherence of of region combinations 
        
        '''
        assert len(low)==len(high) and all([low[i]<=high[i] for i in range(len(low))])
        assert all(len(self.EEG_list[i])==len(self.EEG_list[0]) for i in range(len(self.EEG_list)))
        assert all(self.df_all.loc[i, 'channels']==self.df_all.loc[0, 'channels'] for i in range(len(self.df_all)))
        
        num_channels = len(self.EEG_list[0])
        num_samples = len(self.EEG_list)
        num_bands = len(low)
        coherence = np.zeros( (num_samples, int(num_channels*(num_channels-1)/2), num_bands) )
        win = 5*fs
        
        # Calculate coherence for each signal
        for i_sample in range(len(self.EEG_list)):
            
            for i_comb, (r1, r2) in enumerate(itertools.combinations(range(num_channels), r=2)):
                freqs, C = scipy.signal.coherence(self.EEG_list[i_sample][r1,:], 
                                                  self.EEG_list[i_sample][r2,:], fs=fs, nperseg=win, noverlap=win//2)
                
                # Find intersecting values in frequency vector
                idx = np.logical_and(freqs[:,np.newaxis] >= low, freqs[:,np.newaxis] <= high)
                idx = idx.T   # (65,3)->(3,65)
                
                for i_band in range(idx.shape[0]):
                    coherence[i_sample, i_comb, i_band] = np.mean(C[idx[i_band,:]])
                    
        return coherence.reshape((len(coherence),-1))
                
    def get_covariance(self):
        '''
        Calculate covariance matrix acroos channels for each trial
        
        Return
        -----------
        covariance: 3d array
            i: num_trials
            j, k: num_channels
        '''
        
        assert all(len(self.EEG_list[i])==len(self.EEG_list[0]) for i in range(len(self.EEG_list)))
        
        return np.array([np.cov(EEG) for EEG in self.EEG_list])

def select_features(X_train, X_test, Y_train):
    
    assert isinstance(X_train, np.ndarray) and X_train.ndim==2
    
    sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear'))
    X_train = sel_.fit_transform(X_train, Y_train)
    X_test = sel_.transform(X_test)
    
    return X_train, X_test, sel_
