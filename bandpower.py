#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu June 10 16:44 2021

@author: hundredball
"""
import itertools

import numpy as np
import pandas as pd
import pickle
from scipy import signal
from scipy import interpolate
from scipy.integrate import simps

import dataloader as dl

def get_bandpower(data, fs, low = [4,7,13], high=[7,13,30]):
    '''
    Calculate bandpower of theta, alpha, beta

    Parameters
    ----------
    data : list of 2d array 
        Time signal data
        i : example
        j : channel
        k : sample
    fs : int
        sampling rate

    Returns
    -------
    powers : list of 2d array
        i : example
        j : channel
        k : band

    '''
    assert hasattr(data, '__iter__') and all((sample.ndim==2 for sample in data))
    assert isinstance(fs, int) and fs>0 
    assert hasattr(low, '__iter__') and hasattr(high, '__iter__')
    assert all((high[i]>=low[i]) for i in range(len(high)))
    
    print('Calculating the bandpower of time-series data...')
    # Define window length
    win = 5*fs
    
    num_channel = data[0].shape[0]
    powers = []
    for i, sample in enumerate(data):
        freqs, psd = signal.welch(sample, fs, nperseg=win, noverlap=win/2)
        
        # Frequency resolution
        freq_res = freqs[1] - freqs[0]  # = 1/0.5 = 2
        
        # Find intersecting values in frequency vector
        idx = np.logical_and(freqs[:,np.newaxis] >= low, freqs[:,np.newaxis] <= high)
        idx = idx.T   # (65,3)->(3,65)
        
        # Compute the absolute power by approximating the area under the curve
        sample_powers = np.zeros((num_channel,len(high)))
        for i_band in range(len(high)):
            idx_power = idx[i_band,:]
            sample_powers[:,i_band] = simps(psd[:,idx_power], dx=freq_res)
        
        powers.append(sample_powers)
        
    #print('freqs: ', freqs)
    print('Shape of psd: ', psd.shape)
        
    return powers

if __name__ == '__main__':
    
    # Save data for all subject
    X, Y, _ = dl.read_data()
    powers = get_bandpower(X, 500, low=[3], high=[10])
    print(len(powers))
    